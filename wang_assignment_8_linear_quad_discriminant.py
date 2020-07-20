from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
from collections import deque
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.preprocessing import StandardScaler
from scipy import optimize

def transform_trading_days_to_trading_weeks(df):
    '''
    df: dataframe of relevant data
    returns: dataframe with processed data, only keeping weeks, their open and close for said week
    '''
    trading_list = deque()
    # Iterate through each trading week
    for trading_week, df_trading_week in df.groupby(['Year','Week_Number']):
        classification =  df_trading_week.iloc[0][['Classification']].values[0]
        opening_day_of_week = df_trading_week.iloc[0][['Open']].values[0]
        closing_day_of_week = df_trading_week.iloc[-1][['Close']].values[0]
        trading_list.append([trading_week[0], trading_week[1], opening_day_of_week, closing_day_of_week, classification])
    trading_list_df = pd.DataFrame(np.array(trading_list), columns=['Year', 'Trading Week', 'Week Open', 'Week Close', 'Classification'])
    return trading_list_df

def make_trade(cash, open, close):
    '''
    cash: float of cash on hand
    open: float of open price
    close: float of close price
    returns: The cash made from a long position from open to close
    '''
    shares = np.divide(cash, open)
    return np.multiply(shares, close)

def trading_strategy(trading_df, prediction_label, weekly_balance=100):
    '''
    trading_df: dataframe of relevant weekly data
    prediction_label: the label for which we're going to trade off of
    returns: A df of trades made based on Predicted Labels
    '''
    # The weekly balance we will be using
    weekly_balance_acc = weekly_balance
    trading_history = deque()
    index = 0
    
    while(index < len(trading_df.index) - 1):
        trading_week_index = index
        if weekly_balance_acc != 0:
            # Find the next consecutive green set of weeks and trade on them
            while(trading_week_index < len(trading_df.index) - 1 and trading_df.iloc[trading_week_index][[prediction_label]].values[0] == 'GREEN'):
                trading_week_index += 1
            green_weeks = trading_df.iloc[index:trading_week_index][['Week Open', 'Week Close']]
            # Check if there are green weeks, and if there are not, we add a row for trading history
            if len(green_weeks.index) > 0:
                # Buy shares at open and sell shares at close of week
                green_weeks_open = float(green_weeks.iloc[0][['Week Open']].values[0])
                green_weeks_close = float(green_weeks.iloc[-1][['Week Close']].values[0])
                # We append the money after we make the trade
                weekly_balance_acc = make_trade(weekly_balance_acc, green_weeks_open, green_weeks_close)
            # Regardless of whether we made a trade or not, we append the weekly cash and week over
            trading_history.append([trading_df.iloc[trading_week_index][['Year']].values[0],
                trading_df.iloc[trading_week_index][['Trading Week']].values[0],
                weekly_balance_acc])
        else:
            # If we have no money we will not be able to trade
            trading_history.append([trading_df.iloc[trading_week_index][['Year']].values[0],
                    trading_df.iloc[trading_week_index][['Trading Week']].values[0],
                    weekly_balance_acc])
        index = trading_week_index+1
    trading_hist_df = pd.DataFrame(np.array(trading_history), columns=['Year', 'Trading Week', 'Balance'])
    trading_hist_df['Balance'] = np.round(trading_hist_df[['Balance']].astype(float), 2)

    return trading_hist_df


def main():
    ticker='WMT'
    file_name_self_labels = 'WMT_Labeled_Weeks_Self.csv'
    df = pd.read_csv(file_name_self_labels, encoding='ISO-8859-1')
    df_2018 = df[df['Year'] == 2018]
    df_2019 = df[df['Year'] == 2019]
    # Convert to trading weeks for both years
    trading_weeks_2018 = transform_trading_days_to_trading_weeks(df_2018)
    trading_weeks_2018.reset_index(inplace=True)

    trading_weeks_2019 = transform_trading_days_to_trading_weeks(df_2019)
    trading_weeks_2019.reset_index(inplace=True)

    fit_x_training = StandardScaler().fit_transform(trading_weeks_2018[['Week Close']])
    fix_x_test = StandardScaler().fit_transform(trading_weeks_2019[['Week Close']])
    print('\nQuestion 1:')
    # Linear Discriminant
    lda_classifier = LDA(n_components=2)
    fit_lda = lda_classifier.fit(fit_x_training, trading_weeks_2018[['Classification']].values.ravel())
    print('The equation for the LDA decision boundary is: y = {}x + {}'.format(np.round(fit_lda.coef_.flatten(), 2)[0], np.round(fit_lda.intercept_.flatten(), 2)[0]))
    # Quadratic Discriminant
    qda_classifier = QDA()
    fit_qda = qda_classifier.fit(fit_x_training, trading_weeks_2018[['Classification']].values.ravel())
    # Polyfit the decision function
    fit_x = np.arange(np.min(fit_x_training), np.max(fit_x_training), 0.1)
    fit_y = fit_qda.decision_function(fit_x.reshape(-1,1))
    fitted_params = np.polyfit(fit_x, fit_y, 2)
    print('The equation for the QDA decision boundary is: y = {}x^2 + {}x+ {}'.format(np.round(fitted_params[0], 2), np.round(fitted_params[1], 2),np.round(fitted_params[2], 2)))

    print('\nQuestion 2:')
    y_pred_linear = fit_lda.predict(fix_x_test)
    y_pred_quadtratic = fit_qda.predict(fix_x_test)
    accuracy_lda =  np.round(np.multiply(np.mean(y_pred_linear == trading_weeks_2019[['Classification']].values), 100), 2)
    accuracy_qda =  np.round(np.multiply(np.mean(y_pred_quadtratic == trading_weeks_2019[['Classification']].values), 100), 2)

    print('Accuracy for year 2 LDA: {}%'.format(accuracy_lda))
    print('Accuracy for year 2 QDA: {}%'.format(accuracy_qda))

    print('\nQuestion 3:')
    print('LDA Confusion matrix')
    confusion_matrix_array_lda = confusion_matrix(trading_weeks_2019[['Classification']].values, y_pred_linear)
    confusion_matrix_df_lda = pd.DataFrame(confusion_matrix_array_lda, columns= ['Predicted: GREEN', 'Predicted: RED'], index=['Actual: GREEN', 'Actual: RED'])
    print(confusion_matrix_df_lda)
    print('QDA Confusion matrix')
    confusion_matrix_array_qda = confusion_matrix(trading_weeks_2019[['Classification']].values, y_pred_quadtratic)
    confusion_matrix_df_qda = pd.DataFrame(confusion_matrix_array_qda, columns= ['Predicted: GREEN', 'Predicted: RED'], index=['Actual: GREEN', 'Actual: RED'])
    print(confusion_matrix_df_qda)

    print('\nQuestion 4:')
    print('LDA')
    total_data_points = len(trading_weeks_2019[['Classification']].values)
    true_positive_number = confusion_matrix_df_lda['Predicted: GREEN']['Actual: GREEN']
    true_positive_rate = np.round(np.multiply(np.divide(true_positive_number, total_data_points), 100), 2)
    true_negative_number = confusion_matrix_df_lda['Predicted: RED']['Actual: RED']
    true_negative_rate = np.round(np.multiply(np.divide(true_negative_number, total_data_points), 100), 2)
    print('True positive rate: {}%'.format(true_positive_rate))
    print('True negative rate: {}%'.format(true_negative_rate))

    print('QDA')
    total_data_points = len(trading_weeks_2019[['Classification']].values)
    true_positive_number = confusion_matrix_df_qda['Predicted: GREEN']['Actual: GREEN']
    true_positive_rate = np.round(np.multiply(np.divide(true_positive_number, total_data_points), 100), 2)
    true_negative_number = confusion_matrix_df_qda['Predicted: RED']['Actual: RED']
    true_negative_rate = np.round(np.multiply(np.divide(true_negative_number, total_data_points), 100), 2)
    print('True positive rate: {}%'.format(true_positive_rate))
    print('True negative rate: {}%'.format(true_negative_rate))

    print('\nQuestion 5:')
    buy_and_hold = np.full(len(trading_weeks_2019.index), 'GREEN')
    trading_weeks_2019.insert(len(trading_weeks_2019.columns), "Buy and Hold", buy_and_hold, allow_duplicates=True)
    trading_weeks_2019.insert(len(trading_weeks_2019.columns), "Predicted Labels LDA", y_pred_linear, allow_duplicates=True)
    trading_weeks_2019.insert(len(trading_weeks_2019.columns), "Predicted Labels QDA", y_pred_quadtratic, allow_duplicates=True)
    balance_end_lda = trading_strategy(trading_weeks_2019, "Predicted Labels LDA")[['Balance']].values[-1][0]
    balance_end_qda = trading_strategy(trading_weeks_2019, "Predicted Labels QDA")[['Balance']].values[-1][0]
    balance_end_hold = trading_strategy(trading_weeks_2019, "Buy and Hold")[['Balance']].values[-1][0]

    print('With buy and hold: ${}'.format(balance_end_hold))
    print('With LDA: ${}'.format(balance_end_lda))
    print('With QDA: ${}'.format(balance_end_qda))
    print('Buy and hold is a better strategy.')
    
if __name__ == "__main__":
    main()