"""Symbolic regression over hidden equations using Random Forest."""
# EVOLVE-BLOCK-START
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')


def fit(X_train, y_train):
    """
    Fit a Random Forest Regressor and return a predict callable.
    """
    # Ensure inputs are floats
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    
    # Handle edge case: if y is constant, return that constant
    if np.std(y_train) < 1e-10:
        c = np.mean(y_train)
        def predict_const(X):
            return np.full(len(X), c)
        return predict_const

    # Initialize Random Forest
    # n_estimators=100 is a good balance between accuracy and speed for 200 samples
    # max_depth=None allows the tree to grow until pure or max_leaf_nodes is reached
    # min_samples_leaf=1 allows for very flexible fitting, which is good for symbolic regression
    # where we want to capture exact structures if possible
    # n_jobs=-1 uses all cores
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=None,
        n_jobs=-1,
        random_state=42
    )
    
    rf.fit(X_train, y_train)
    
    def predict(X):
        X = np.asarray(X, dtype=np.float64)
        return rf.predict(X)
        
    return predict
# EVOLVE-BLOCK-END
