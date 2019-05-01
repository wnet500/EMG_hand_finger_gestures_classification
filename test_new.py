import pandas as pd
import numpy as np
import mglearn # 그래프
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier # 모델
from sklearn.metrics import confusion_matrix # 오차행렬
from sklearn.preprocessing import StandardScaler # 표준화
from sklearn.model_selection import GridSearchCV # 교차검증
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

from IPython.core.display import Image, display
print('계층별 임의 분할 교차 검증 개념 그림: ')
display(Image(filename='StratifiedShuffleSplit.jpg'))

# Load Data
X_data = np.loadtxt('./t_X_data_1.txt', dtype=np.float32, delimiter=', ')
y_data = np.loadtxt('./t_Y_data_1.txt', dtype=np.int32, delimiter=', ')

# StandardScaler: 평균0, 분산1
scaler = StandardScaler()
scaler.fit(X_data)
X_data_scaled = scaler.transform(X_data)

# Parameter
param_grid = [{'hidden_layer_sizes': [[120], [160], [200], [240], [280], [320], [360], [400], [440], [480], [520], [560], [600]],
               'alpha': [0.01, 0.1, 1, 10],
               'max_iter': [1000, 1500, 2000, 2500, 3000]}]
#print('매개변수 그리드: \n{}'.format(param_grid))

# Grid Search
stratified_shuffle_split = StratifiedShuffleSplit(n_splits=5, test_size=0.25, train_size=0.75, random_state=0)

grid_search = GridSearchCV(MLPClassifier(solver='lbfgs', random_state=0),
                           param_grid, cv=stratified_shuffle_split,
                           n_jobs=-1, return_train_score=True) #모든 Core사용

#X_train, X_test, y_train, y_test = train_test_split(X_data_scaled, y_data, random_state=42) #test_size = 0.25 (defalt)

grid_search.fit(X_data, y_data)

# Load Data
X_data_test = np.loadtxt('./X_data_1.txt', dtype=np.float32, delimiter=', ')
y_data_test = np.loadtxt('./Y_data_1.txt', dtype=np.int32, delimiter=', ')

X_data_test = scaler.transform(X_data_test)

print('테스트 세트 점수: {:.2f}'.format(grid_search.score(X_data_test, y_data_test)))
print('최적 매개변수: {}'.format(grid_search.best_params_))
print('최상 교차 검증 점수: {:.2f}'.format(grid_search.best_score_))

# Result
results = pd.DataFrame(grid_search.cv_results_)
results.to_excel("ELM_Gridsearch_results.xlsx")
display(results.T)
#display(results.head())
