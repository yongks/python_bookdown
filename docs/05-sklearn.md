# sklearn
This is a machine learning library.

## Setup (hidden)




```python
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:75% !important; margin-left:350px; }</style>"))
#matplotlib inline
```

```
## <IPython.core.display.HTML object>
```

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
pd.set_option( 'display.notebook_repr_html', False)  # render Series and DataFrame as text, not HTML
pd.set_option( 'display.max_column', 10)    # number of columns
pd.set_option( 'display.max_rows', 10)     # number of rows
pd.set_option( 'display.width', 90)        # number of characters per row

import os
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = "C:\ProgramData\Anaconda3\Library\plugins\platforms"
#import matplotlib
#matplotlib.use('Qt5Agg')
```

## The Library
sklearn **does not automatically import its subpackages**. Therefore all subpakcages must be specifically loaded before use.


```python
# Sample Data
from sklearn                 import datasets

# Model Selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_validate

# Preprocessing
from sklearn.impute   import SimpleImputer
from sklearn.preprocessing   import MinMaxScaler
from sklearn.preprocessing   import StandardScaler
from sklearn.preprocessing   import Normalizer
from sklearn.preprocessing   import PolynomialFeatures

# Model and Pipeline
from sklearn.linear_model    import LinearRegression,Lasso
from sklearn.pipeline        import make_pipeline

# Measurement
from sklearn.metrics         import *

import statsmodels.formula.api as smf
```

## Model Fitting


![split](img/fitting.png)


### Underfitting
- The model does not fit the training data and therefore misses the trends in the data
- The model cannot be generalized to new data, this is usually the result of a very simple model (not enough predictors/independent variables)
- The model will have poor predictive ability
- For example, we fit a linear model (like linear regression) to data that is not linear


### Overfitting
- The model has trained ?too well? and is now, well, fit too closely to the training dataset
- The model is too complex (i.e. **too many features/variables** compared to the number of observations)
- The model will be very accurate on the training data but will probably be very not accurate on untrained or new data
- The model is not generalized (or not AS generalized), meaning you can generalize the results 
- The model learns or describes the ?noise? in the training data instead of the actual relationships between variables in the data


### Just Right
- It is worth noting the underfitting is not as prevalent as overfitting
- Nevertheless, we want to avoid both of those problems in data analysis
- We want to find the middle ground between under and overfitting our model


## Model Tuning
- A highly complex model tend to overfit
- A too flexible model tend to underfit

Complexity can be reduced by: 
- Less features
- Less degree of polynomial features
- Apply generalization (tuning hyperparameters)


![split](img/model_complexity.jpg)


## High Level ML Process


![split](img/scikit_learn_split.jpg)


## Built-in Datasets
sklearn included some popular datasets to play with  
Each dataset is of type **Bunch**.   
It has useful data (array) in the form of properties:  
- keys (display all data availabe within the dataset)  
- data (common)  
- target (common)  
- DESCR (common)
- feature_names (some dataset)  
- target_names  (some dataset) 
- images (some dataset)


### diabetes (regression)


#### Load Dataset


```python
diabetes = datasets.load_diabetes()
print (type(diabetes))
```

```
## <class 'sklearn.utils._bunch.Bunch'>
```


#### keys


```python
diabetes.keys()
```

```
## dict_keys(['data', 'target', 'frame', 'DESCR', 'feature_names', 'data_filename', 'target_filename', 'data_module'])
```


#### Features and Target
.data   = features  - two dimension array  
.target = target    - one dimension array 


```python
print (type(diabetes.data))
```

```
## <class 'numpy.ndarray'>
```

```python
print (type(diabetes.target))
```

```
## <class 'numpy.ndarray'>
```

```python
print (diabetes.data.shape)
```

```
## (442, 10)
```

```python
print (diabetes.target.shape)
```

```
## (442,)
```


#### Load with X,y (Convenient Method)
using return_X_y = True, data is loaded into X, target is loaded into y


```python
X,y      = datasets.load_diabetes(return_X_y=True)
```


```python
print (X.shape)
```

```
## (442, 10)
```

```python
print (y.shape)
```

```
## (442,)
```


### digits (Classification)
This is a copy of the test set of the UCI ML hand-written digits datasets


```python
digits = datasets.load_digits()
print (type(digits))
```

```
## <class 'sklearn.utils._bunch.Bunch'>
```

```python
print (type(digits.data))
```

```
## <class 'numpy.ndarray'>
```


```python
digits.keys()
```

```
## dict_keys(['data', 'target', 'frame', 'feature_names', 'target_names', 'images', 'DESCR'])
```


```python
digits.target_names
```

```
## array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
```


#### data


```python
digits.data.shape  # features
```

```
## (1797, 64)
```


```python
digits.target.shape # target
```

```
## (1797,)
```


#### Images
- images is 3 dimensional array  
- There are **1797 samples**, each sample is **8x8 pixels**


```python
digits.images.shape
```

```
## (1797, 8, 8)
```


```python
type(digits.images)
```

```
## <class 'numpy.ndarray'>
```


**Each element represent the data that make its target**


```python
print (digits.target[100])
```

```
## 4
```

```python
print (digits.images[100])
```

```
## [[ 0.  0.  0.  2. 13.  0.  0.  0.]
##  [ 0.  0.  0.  8. 15.  0.  0.  0.]
##  [ 0.  0.  5. 16.  5.  2.  0.  0.]
##  [ 0.  0. 15. 12.  1. 16.  4.  0.]
##  [ 0.  4. 16.  2.  9. 16.  8.  0.]
##  [ 0.  0. 10. 14. 16. 16.  4.  0.]
##  [ 0.  0.  0.  0. 13.  8.  0.  0.]
##  [ 0.  0.  0.  0. 13.  6.  0.  0.]]
```

```python
plt.matshow(digits.images[100]) 
```

<img src="05-sklearn_files/figure-html/unnamed-chunk-16-1.png" width="480" />


#### Loading Into X,y (Convenient Method)


```python
X,y = datasets.load_digits(return_X_y=True)
```


```python
X.shape
```

```
## (1797, 64)
```


```python
y.shape
```

```
## (1797,)
```


### iris (Classification)


```python
iris = datasets.load_iris()
```


```python
iris.keys()
```

```
## dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename', 'data_module'])
```


#### Feature Names


```python
iris.feature_names
```

```
## ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
```


#### target


```python
iris.target_names
```

```
## array(['setosa', 'versicolor', 'virginica'], dtype='<U10')
```


```python
iris.target
```

```
## array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
##        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
##        0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
##        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
##        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
##        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
##        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
```


## Train Test Data Splitting


### Sample Data
Generate 100 rows of data, with 3x features (X1,X2,X3), and one dependant variable (Y)


```python
n = 21  # number of samples
I = 5  # intercept value
E = np.random.randint( 1,20, n)  # Error
x1 = np.random.randint( 1,n+1, n)
x2 = np.random.randint( 1,n+1, n)
x3 = np.random.randint( 1,n+1, n)
y = 0.1*x1 + 0.2*x2 + 0.3*x3 + E + I
mydf = pd.DataFrame({
    'y':y,
    'x1':x1,
    'x2':x2,
    'x3':x3
})
mydf.shape
```

```
## (21, 4)
```


### One Time Split

sklearn::train_test_split() has two forms:
- Take one DF, split into 2 DF (most of sklearn modeling use this method
- Take two DFs, split into 4 DF


```python
mydf.head()
```

```
##       y  x1  x2  x3
## 0  29.7   4  18  19
## 1  21.4   9  19  19
## 2  29.8   1   5  19
## 3  31.3   7  14  16
## 4  19.3  11  16  20
```


#### Method 1: Split One Dataframe Into Two (Train & Test)

```
traindf, testdf = train_test_split( df, test_size=, random_state= ) 
 # random_state : seed number (integer), optional
 # test_size    : fraction of 1, 0.2 means 20%
```

![split](img/train_test_split_1.jpg)


```python
traindf, testdf = train_test_split(mydf,test_size=0.2, random_state=25)
```


```python
print (len(traindf))
```

```
## 16
```

```python
print (len(testdf))
```

```
## 5
```


#### Method 2: Split Two DataFrame (X,Y) into Four x_train/test, y_train/test
```
x_train, x_test, y_train, y_test = train_test_split( X,Y, test_size=, random_state= )
 # random_state : seed number (integer), optional
 # test_size    : fraction of 1, 0.2 means 20%
```

![split](img/train_test_split_2.jpg)


**Split DataFrame into X and Y First**


```python
feature_cols = ['x1','x2','x3']
X = mydf[feature_cols]
Y = mydf.y
```


**Then Split X/Y into x_train/test, y_train/test**


```python
x_train, x_test, y_train, y_test = train_test_split( X,Y, test_size=0.2, random_state=25)
print (len(x_train))
```

```
## 16
```

```python
print (len(x_test))
```

```
## 5
```


### K-Fold
```
KFold(n_splits=3, shuffle=False, random_state=None)
```

![split](img/kfold.jpg)


**suffle=False** (default), meaning index number is taken continously


```python
kf = KFold(n_splits=7)
```


```python
for train_index, test_index in kf.split(X):
  print (train_index, test_index)
```

```
## [ 3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20] [0 1 2]
## [ 0  1  2  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20] [3 4 5]
## [ 0  1  2  3  4  5  9 10 11 12 13 14 15 16 17 18 19 20] [6 7 8]
## [ 0  1  2  3  4  5  6  7  8 12 13 14 15 16 17 18 19 20] [ 9 10 11]
## [ 0  1  2  3  4  5  6  7  8  9 10 11 15 16 17 18 19 20] [12 13 14]
## [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 18 19 20] [15 16 17]
## [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17] [18 19 20]
```


**shuffle=True**


```python
kf = KFold(n_splits=7, shuffle=True)
```


```python
for train_index, test_index in kf.split(X):
  print (train_index, test_index)
```

```
## [ 0  1  2  3  5  6  7  8  9 10 11 12 13 14 15 16 17 20] [ 4 18 19]
## [ 0  1  2  4  5  6  7  8 10 11 12 13 14 15 17 18 19 20] [ 3  9 16]
## [ 0  1  3  4  5  7  8  9 10 11 12 13 15 16 17 18 19 20] [ 2  6 14]
## [ 0  2  3  4  5  6  8  9 10 11 12 13 14 15 16 18 19 20] [ 1  7 17]
## [ 0  1  2  3  4  5  6  7  9 10 11 12 14 15 16 17 18 19] [ 8 13 20]
## [ 1  2  3  4  5  6  7  8  9 10 11 13 14 16 17 18 19 20] [ 0 12 15]
## [ 0  1  2  3  4  6  7  8  9 12 13 14 15 16 17 18 19 20] [ 5 10 11]
```


### Leave One Out

- For a dataset of N rows, Leave One Out will split N-1 times, each time leaving one row as test, remaning as training set.  
- Due to the **high number of test sets** (which is the same as the number of samples-1) this cross-validation method can be very costly. For large datasets one should favor KFold.


```python
loo = LeaveOneOut()
```


```python
for train_index, test_index in loo.split(X):
  print (train_index, test_index)
```

```
## [ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20] [0]
## [ 0  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20] [1]
## [ 0  1  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20] [2]
## [ 0  1  2  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20] [3]
## [ 0  1  2  3  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20] [4]
## [ 0  1  2  3  4  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20] [5]
## [ 0  1  2  3  4  5  7  8  9 10 11 12 13 14 15 16 17 18 19 20] [6]
## [ 0  1  2  3  4  5  6  8  9 10 11 12 13 14 15 16 17 18 19 20] [7]
## [ 0  1  2  3  4  5  6  7  9 10 11 12 13 14 15 16 17 18 19 20] [8]
## [ 0  1  2  3  4  5  6  7  8 10 11 12 13 14 15 16 17 18 19 20] [9]
## [ 0  1  2  3  4  5  6  7  8  9 11 12 13 14 15 16 17 18 19 20] [10]
## [ 0  1  2  3  4  5  6  7  8  9 10 12 13 14 15 16 17 18 19 20] [11]
## [ 0  1  2  3  4  5  6  7  8  9 10 11 13 14 15 16 17 18 19 20] [12]
## [ 0  1  2  3  4  5  6  7  8  9 10 11 12 14 15 16 17 18 19 20] [13]
## [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 15 16 17 18 19 20] [14]
## [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 16 17 18 19 20] [15]
## [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 17 18 19 20] [16]
## [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 18 19 20] [17]
## [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 19 20] [18]
## [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 20] [19]
## [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19] [20]
```


```python
X
```

```
##     x1  x2  x3
## 0    4  18  19
## 1    9  19  19
## 2    1   5  19
## 3    7  14  16
## 4   11  16  20
## ..  ..  ..  ..
## 16  11   6   5
## 17   1  16  17
## 18  18  21   7
## 19  16  14   9
## 20  21   7   1
## 
## [21 rows x 3 columns]
```


## Polynomial Transform
This can be used as part of feature engineering, to introduce new features for data that seems to fit with quadradic model.


### Single Variable


#### Sample Data
Data must be 2-D before polynomial features can be applied. Code below convert 1D array into 2D array.


```python
x = np.array([1, 2, 3, 4, 5])
X = x[:,np.newaxis]
X
```

```
## array([[1],
##        [2],
##        [3],
##        [4],
##        [5]])
```


#### Degree 1
One Degree means maintain original features. No new features is created.


```python
PolynomialFeatures(degree=1, include_bias=False).fit_transform(X)
```

```
## array([[1.],
##        [2.],
##        [3.],
##        [4.],
##        [5.]])
```


#### Degree 2
Degree-1 original   feature:  x  
Degree-2 additional features:  x^2  


```python
PolynomialFeatures(degree=2, include_bias=False).fit_transform(X)
```

```
## array([[ 1.,  1.],
##        [ 2.,  4.],
##        [ 3.,  9.],
##        [ 4., 16.],
##        [ 5., 25.]])
```


#### Degree 3
Degree-1 original   feature:  x  
Degree-2 additional features:  x^2  
Degree-3 additional features:  x^3


```python
PolynomialFeatures(degree=3, include_bias=False).fit_transform(X)
```

```
## array([[  1.,   1.,   1.],
##        [  2.,   4.,   8.],
##        [  3.,   9.,  27.],
##        [  4.,  16.,  64.],
##        [  5.,  25., 125.]])
```


#### Degree 4
Degree-1 original   feature:  x  
Degree-2 additional features:  x^2  
Degree-3 additional features:  x^3  
Degree-3 additional features:  x^4


```python
PolynomialFeatures(degree=4, include_bias=False).fit_transform(X)
```

```
## array([[  1.,   1.,   1.,   1.],
##        [  2.,   4.,   8.,  16.],
##        [  3.,   9.,  27.,  81.],
##        [  4.,  16.,  64., 256.],
##        [  5.,  25., 125., 625.]])
```


### Two Variables


#### Sample Data


```python
X = pd.DataFrame( {'x1': [1, 2, 3, 4, 5 ],
                   'x2': [6, 7, 8, 9, 10]})
X
```

```
##    x1  x2
## 0   1   6
## 1   2   7
## 2   3   8
## 3   4   9
## 4   5  10
```


#### Degree 2
```
Degree-1 original   features:  x1,     x2  
Degree-2 additional features:  x1^2,   x2^2,   x1:x2 
```


```python
PolynomialFeatures(degree=2, include_bias=False).fit_transform(X)
```

```
## array([[  1.,   6.,   1.,   6.,  36.],
##        [  2.,   7.,   4.,  14.,  49.],
##        [  3.,   8.,   9.,  24.,  64.],
##        [  4.,   9.,  16.,  36.,  81.],
##        [  5.,  10.,  25.,  50., 100.]])
```


#### Degree 3
```
Degree-1 original   features:  x1,       x2  
Degree-2 additional features:  x1^2,     x2^2,   x1:x2 
Degree-3 additional features:  x1^3,     x2^3    x1:x2^2    x2:x1^2
```


```python
PolynomialFeatures(degree=3, include_bias=False).fit_transform(X)
```

```
## array([[   1.,    6.,    1.,    6.,   36.,    1.,    6.,   36.,  216.],
##        [   2.,    7.,    4.,   14.,   49.,    8.,   28.,   98.,  343.],
##        [   3.,    8.,    9.,   24.,   64.,   27.,   72.,  192.,  512.],
##        [   4.,    9.,   16.,   36.,   81.,   64.,  144.,  324.,  729.],
##        [   5.,   10.,   25.,   50.,  100.,  125.,  250.,  500., 1000.]])
```


## Imputation of Missing Data


### Sample Data


```python
from numpy import nan
X = np.array([[ nan, 0,   3  ],
              [ 3,   7,   9  ],
              [ 3,   5,   2  ],
              [ 4,   nan, 6  ],
              [ 8,   8,   1  ]])

y = np.array([14, 16, -1,  8, -5])
```


### Imputer


#### mean strategy


```python
imp = SimpleImputer(strategy='mean')
X2 = imp.fit_transform(X)
X2
```

```
## array([[4.5, 0. , 3. ],
##        [3. , 7. , 9. ],
##        [3. , 5. , 2. ],
##        [4. , 5. , 6. ],
##        [8. , 8. , 1. ]])
```


## Scaling

It is possible that some insignificant variable with larger range will be dominating the objective function.  
We can remove this problem by scaling down all the features to a same range.

### Sample Data


```python
X=mydf.filter(like='x')[:5]
X
```

```
##    x1  x2  x3
## 0   4  18  19
## 1   9  19  19
## 2   1   5  19
## 3   7  14  16
## 4  11  16  20
```


### MinMax Scaler

```
MinMaxScaler( feature_range(0,1), copy=True )
# default feature range (output result) from 0 to 1
# default return a copy of new array, copy=False will inplace original array
```


**Define Scaler Object**


```python
scaler = MinMaxScaler()
```


**Transform Data**


```python
scaler.fit_transform(X)
```

```
## array([[0.3       , 0.92857143, 0.75      ],
##        [0.8       , 1.        , 0.75      ],
##        [0.        , 0.        , 0.75      ],
##        [0.6       , 0.64285714, 0.        ],
##        [1.        , 0.78571429, 1.        ]])
```


**Scaler Attributes**  
```
data_min_: minimum value of the feature (before scaling)  
data_max_: maximum value of the feature (before scaling)  
```


```python
pd.DataFrame(list(zip(scaler.data_min_, scaler.data_max_)), 
             columns=['data_min','data_max'], 
             index=X.columns)
```

```
##     data_min  data_max
## x1       1.0      11.0
## x2       5.0      19.0
## x3      16.0      20.0
```


### Standard Scaler

It is most suitable for techniques that assume a Gaussian distribution in the input variables and work better with rescaled data, such as linear regression, logistic regression and linear discriminate analysis.

```
StandardScaler(copy=True, with_mean=True, with_std=True)
# copy=True : return a copy of data, instead of inplace
# with_mean=True : centre all features by substracting with its mean
# with_std=True  : centre all features by dividing with its std
```


**Define Scaler Object**


```python
scaler = StandardScaler()
```


**Transform Data**


```python
scaler.fit_transform(X)
```

```
## array([[-0.67505274,  0.71942469,  0.29488391],
##        [ 0.73130714,  0.91926488,  0.29488391],
##        [-1.51886867, -1.8784978 ,  0.29488391],
##        [ 0.16876319, -0.07993608, -1.91674543],
##        [ 1.29385109,  0.31974431,  1.03209369]])
```


**Scaler Attributes**  
After the data transformation step above, scaler will have the mean and variance information for each feature.


```python
pd.DataFrame(list(zip(scaler.mean_, scaler.var_)), 
             columns=['mean','variance'], 
             index=X.columns)
```

```
##     mean  variance
## x1   6.4     12.64
## x2  14.4     25.04
## x3  18.6      1.84
```


## Pipeline


With any of the preceding examples, it can quickly become tedious to do the transformations by hand, especially if you wish to string together multiple steps. For example, we might want a processing pipeline that looks something like this:

- **Impute** missing values using the mean  
- **Transform** features to quadratic  
- **Fit** a linear regression  

**make_pipeline** takes list of functions as parameters. When calling **fit()** on a pipeline object, these functions will be performed in sequential with data flow from one function to another.

```
make_pipeline (
    function_1 (),
    function_2 (),
    function_3 ()
 )
```


### Sample Data


```python
X
```

```
##    x1  x2  x3
## 0   4  18  19
## 1   9  19  19
## 2   1   5  19
## 3   7  14  16
## 4  11  16  20
```


```python
y
```

```
## array([14, 16, -1,  8, -5])
```


### Create Pipeline


```python
my_pipe = make_pipeline (
    SimpleImputer            (strategy='mean'),
    PolynomialFeatures (degree=2),
    LinearRegression   ()
)
type(my_pipe)
```

```
## <class 'sklearn.pipeline.Pipeline'>
```


```python
my_pipe
```

```{=html}
<style>#sk-container-id-1 {color: black;background-color: white;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>Pipeline(steps=[(&#x27;simpleimputer&#x27;, SimpleImputer()),
                (&#x27;polynomialfeatures&#x27;, PolynomialFeatures()),
                (&#x27;linearregression&#x27;, LinearRegression())])</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" ><label for="sk-estimator-id-1" class="sk-toggleable__label sk-toggleable__label-arrow">Pipeline</label><div class="sk-toggleable__content"><pre>Pipeline(steps=[(&#x27;simpleimputer&#x27;, SimpleImputer()),
                (&#x27;polynomialfeatures&#x27;, PolynomialFeatures()),
                (&#x27;linearregression&#x27;, LinearRegression())])</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-2" type="checkbox" ><label for="sk-estimator-id-2" class="sk-toggleable__label sk-toggleable__label-arrow">SimpleImputer</label><div class="sk-toggleable__content"><pre>SimpleImputer()</pre></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-3" type="checkbox" ><label for="sk-estimator-id-3" class="sk-toggleable__label sk-toggleable__label-arrow">PolynomialFeatures</label><div class="sk-toggleable__content"><pre>PolynomialFeatures()</pre></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-4" type="checkbox" ><label for="sk-estimator-id-4" class="sk-toggleable__label sk-toggleable__label-arrow">LinearRegression</label><div class="sk-toggleable__content"><pre>LinearRegression()</pre></div></div></div></div></div></div></div>
```


### Executing Pipeline


```python
my_pipe.fit( X, y) # execute the pipeline
```

```{=html}
<style>#sk-container-id-2 {color: black;background-color: white;}#sk-container-id-2 pre{padding: 0;}#sk-container-id-2 div.sk-toggleable {background-color: white;}#sk-container-id-2 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-2 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-2 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-2 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-2 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-2 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-2 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-2 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-2 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-2 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-2 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-2 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-2 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-2 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-2 div.sk-item {position: relative;z-index: 1;}#sk-container-id-2 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-2 div.sk-item::before, #sk-container-id-2 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-2 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-2 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-2 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-2 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-2 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-2 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-2 div.sk-label-container {text-align: center;}#sk-container-id-2 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-2 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-2" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>Pipeline(steps=[(&#x27;simpleimputer&#x27;, SimpleImputer()),
                (&#x27;polynomialfeatures&#x27;, PolynomialFeatures()),
                (&#x27;linearregression&#x27;, LinearRegression())])</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-5" type="checkbox" ><label for="sk-estimator-id-5" class="sk-toggleable__label sk-toggleable__label-arrow">Pipeline</label><div class="sk-toggleable__content"><pre>Pipeline(steps=[(&#x27;simpleimputer&#x27;, SimpleImputer()),
                (&#x27;polynomialfeatures&#x27;, PolynomialFeatures()),
                (&#x27;linearregression&#x27;, LinearRegression())])</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-6" type="checkbox" ><label for="sk-estimator-id-6" class="sk-toggleable__label sk-toggleable__label-arrow">SimpleImputer</label><div class="sk-toggleable__content"><pre>SimpleImputer()</pre></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-7" type="checkbox" ><label for="sk-estimator-id-7" class="sk-toggleable__label sk-toggleable__label-arrow">PolynomialFeatures</label><div class="sk-toggleable__content"><pre>PolynomialFeatures()</pre></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-8" type="checkbox" ><label for="sk-estimator-id-8" class="sk-toggleable__label sk-toggleable__label-arrow">LinearRegression</label><div class="sk-toggleable__content"><pre>LinearRegression()</pre></div></div></div></div></div></div></div>
```

```python
print (y)
```

```
## [14 16 -1  8 -5]
```

```python
print (my_pipe.predict(X))
```

```
## [14. 16. -1.  8. -5.]
```


```python
type(my_pipe)
```

```
## <class 'sklearn.pipeline.Pipeline'>
```


## Cross Validation


### Load Data


```python
X,y = datasets.load_diabetes(return_X_y=True)
```


### Choose An Cross Validator


```python
kf = KFold(n_splits=5)
```


### Run Cross Validation


**Single Scorer**  
Use default scorer of the estimator (if available)


```python
lasso = Lasso()
cv_results1 = cross_validate(lasso, X,y,cv=kf,
    return_train_score=False)
```


**Multiple Scorer**  
Specify the scorer
http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter


```python
cv_results2 = cross_validate(lasso, X,y,cv=kf,
    scoring=("neg_mean_absolute_error","neg_mean_squared_error","r2"),
    return_train_score=False)
```


### The Result
Result is a **dictionary**


```python
cv_results1.keys()
```

```
## dict_keys(['fit_time', 'score_time', 'test_score'])
```


```python
cv_results2.keys()
```

```
## dict_keys(['fit_time', 'score_time', 'test_neg_mean_absolute_error', 'test_neg_mean_squared_error', 'test_r2'])
```


```python
cv_results1
```

```
## {'fit_time': array([0.        , 0.01567507, 0.        , 0.        , 0.        ]), 'score_time': array([0., 0., 0., 0., 0.]), 'test_score': array([0.28349006, 0.35157952, 0.35338233, 0.33481253, 0.36453239])}
```


```python
cv_results2
```

```
## {'fit_time': array([0.        , 0.        , 0.        , 0.01587486, 0.        ]), 'score_time': array([0.        , 0.        , 0.        , 0.00050807, 0.        ]), 'test_neg_mean_absolute_error': array([-50.09006473, -52.54118496, -55.02819607, -50.8112893 ,
##        -55.60479053]), 'test_neg_mean_squared_error': array([-3491.74208572, -4113.86049974, -4046.91135088, -3489.75176794,
##        -4111.92674103]), 'test_r2': array([0.28349006, 0.35157952, 0.35338233, 0.33481253, 0.36453239])}
```
