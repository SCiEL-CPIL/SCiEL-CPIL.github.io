# EVOLVE-BLOCK-START
import numpy as np
from scipy.interpolate import PchipInterpolator, CubicSpline
import time

def fit(X_train, y_train):
    """
    Fits a 1D interpolating spline to noiseless data.
    Since data is noiseless and dense (200 points), interpolation
    should yield near-zero error on test points within the training range.
    """
    X = np.asarray(X_train, dtype=np.float64).flatten()
    y = np.asarray(y_train, dtype=np.float64).flatten()
    
    # Ensure 1D
    X = X.reshape(-1)
    y = y.reshape(-1)
    
    # Sort by X
    sort_idx = np.argsort(X)
    X_sorted = X[sort_idx]
    y_sorted = y[sort_idx]
    
    # Check for duplicate X values and average y if needed
    _, unique_indices = np.unique(X_sorted, return_index=True)
    if len(unique_indices) < len(X_sorted):
        X_unique = X_sorted[unique_indices]
        y_unique = np.array([y_sorted[X_sorted == xv].mean() for xv in X_unique])
    else:
        X_unique = X_sorted
        y_unique = y_sorted
    
    # Try PCHIP first (monotonicity-preserving, no overshoot)
    try:
        interp = PchipInterpolator(X_unique, y_unique)
        # Verify interpolation on training data
        y_pred_train = interp(X_unique)
        train_mse = np.mean((y_pred_train - y_unique)**2)
        
        # Also try cubic spline as backup
        cs = CubicSpline(X_unique, y_unique)
        y_pred_train_cs = cs(X_unique)
        train_mse_cs = np.mean((y_pred_train_cs - y_unique)**2)
        
        # Choose the one with lower training MSE (should be near zero for both)
        if train_mse <= train_mse_cs:
            def predict(X_test):
                X_test = np.asarray(X_test, dtype=np.float64).flatten()
                return interp(X_test)
        else:
            def predict(X_test):
                X_test = np.asarray(X_test, dtype=np.float64).flatten()
                return cs(X_test)
        
        return predict
    except Exception:
        # Fallback to linear interpolation
        def predict(X_test):
            X_test = np.asarray(X_test, dtype=np.float64).flatten()
            return np.interp(X_test, X_unique, y_unique)
        return predict
# EVOLVE-BLOCK-END
