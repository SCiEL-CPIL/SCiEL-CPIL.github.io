"""Symbolic regression via Ensemble of Random Forests with diverse hyperparameters."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def fit(X_train, y_train):
    start_time = time.time()
    DEADLINE = 325.0  # Safety margin for 330s limit
    
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    
    n, d = X.shape
    
    # Normalize inputs to [0, 1] for better tree splitting behavior
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X_range = X_max - X_min
    X_range[X_range == 0] = 1.0
    X_norm = (X - X_min) / X_range
    
    # Define a set of diverse Random Forest configurations
    # We vary n_estimators, max_depth, min_samples_split, and random_state
    configs = [
        {'n_estimators': 50, 'max_depth': 10, 'min_samples_split': 2, 'random_state': 42},
        {'n_estimators': 100, 'max_depth': 15, 'min_samples_split': 2, 'random_state': 43},
        {'n_estimators': 200, 'max_depth': 20, 'min_samples_split': 2, 'random_state': 44},
        {'n_estimators': 100, 'max_depth': 8, 'min_samples_split': 5, 'random_state': 45},
        {'n_estimators': 150, 'max_depth': 12, 'min_samples_split': 3, 'random_state': 46},
        {'n_estimators': 300, 'max_depth': None, 'min_samples_leaf': 2, 'random_state': 47},
        {'n_estimators': 200, 'max_depth': 25, 'min_samples_split': 2, 'random_state': 48},
        {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 4, 'random_state': 49},
    ]
    
    models = []
    weights = []
    
    # Train each model in the ensemble, respecting the time deadline
    for config in configs:
        if time.time() > DEADLINE:
            break
            
        try:
            # Use a subset of time for fitting to allow more models if possible
            # But RF is generally fast. Let's just fit.
            rf = RandomForestRegressor(**config)
            
            # Record time before fit
            t0 = time.time()
            rf.fit(X_norm, y)
            t_fit = time.time() - t0
            
            # Evaluate on training data to get a proxy for quality (MSE)
            # Since we don't have a separate validation set, we use training MSE
            # But training MSE is often low. Better to use a small held-out set?
            # With 200 points, we can hold out 20 for validation.
            # Let's do a quick K-fold style check or just trust the diversity.
            # Actually, let's compute training MSE for weighting. Lower is better.
            y_pred_train = rf.predict(X_norm)
            mse_train = mean_squared_error(y, y_pred_train)
            
            # Add model to ensemble
            models.append(rf)
            # Use inverse MSE as weight (add small epsilon to avoid div by zero)
            w = 1.0 / (mse_train + 1e-6)
            weights.append(w)
            
        except Exception as e:
            # If a model fails, skip it
            continue
    
    # If no models were trained (unlikely), fall back to a simple linear regression
    if len(models) == 0:
        def predict_fallback(X_test):
            X_test = np.asarray(X_test, dtype=np.float64)
            X_test_norm = (X_test - X_min) / X_range
            cols = [np.ones(len(X_test))]
            for j in range(d):
                cols.append(X_test_norm[:, j])
            feat = np.column_stack(cols)
            w_ls, *_ = np.linalg.lstsq(np.column_stack([np.ones(n)] + [X_norm[:, j] for j in range(d)]), y, rcond=None)
            return feat @ w_ls
        return predict_fallback
    
    # Normalize weights
    weights = np.array(weights)
    weights = weights / np.sum(weights)
    
    def predict(X_test):
        X_test = np.asarray(X_test, dtype=np.float64)
        X_test_norm = (X_test - X_min) / X_range
        
        predictions = np.zeros((len(X_test), len(models)))
        
        for i, model in enumerate(models):
            try:
                predictions[:, i] = model.predict(X_test_norm)
            except Exception:
                # If prediction fails for some reason, use 0 (or mean y)
                predictions[:, i] = np.mean(y)
        
        # Weighted average
        pred = predictions @ weights
        
        return pred

    return predict
# EVOLVE-BLOCK-END
