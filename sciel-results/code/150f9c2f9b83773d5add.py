import numpy as np
import time
from scipy.interpolate import CubicSpline

def fit(X_train, y_train):
    """
    Fit a regression model for 1D noiseless data using cubic spline interpolation.
    """
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    
    # Flatten if 2D with 1 column
    if X.ndim == 2 and X.shape[1] == 1:
        x = X.flatten()
    elif X.ndim == 1:
        x = X
    else:
        # Fallback for multi-dim: use first column or PCA? 
        # Contract says 1 input, so this shouldn't happen.
        x = X[:, 0] if X.ndim == 2 else X
        
    # Sort by x
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]
    
    # Remove duplicate x values if any (take mean of y, though noiseless implies unique)
    unique_x, unique_indices = np.unique(x_sorted, return_index=True)
    if len(unique_x) < len(x_sorted):
        # Average y values for duplicate x
        y_unique = np.zeros_like(unique_x)
        for i, idx in enumerate(unique_indices):
            mask = (x_sorted == unique_x[i])
            y_unique[i] = np.mean(y_sorted[mask])
        x_sorted = unique_x
        y_sorted = y_unique

    # Fit Cubic Spline
    # bc_type='natural' assumes second derivative is 0 at boundaries, which might not be true.
    # 'not-a-knot' is often better for internal smoothness.
    try:
        spline = CubicSpline(x_sorted, y_sorted, bc_type='not-a-knot')
    except Exception:
        # Fallback to linear if spline fails (e.g., not enough points)
        from scipy.interpolate import interp1d
        if len(x_sorted) >= 2:
            spline = interp1d(x_sorted, y_sorted, kind='linear', fill_value='extrapolate')
        else:
            y_mean = np.mean(y_sorted)
            def predict_const(X_test):
                X_test = np.asarray(X_test, dtype=np.float64)
                return np.full(X_test.shape[0], y_mean)
            return predict_const

    def predict(X_test):
        X_test = np.asarray(X_test, dtype=np.float64)
        if X_test.ndim == 2 and X_test.shape[1] == 1:
            x_test = X_test.flatten()
        elif X_test.ndim == 1:
            x_test = X_test
        else:
            x_test = X_test[:, 0]
            
        pred = spline(x_test)
        
        # Ensure finite output
        # If extrapolation causes huge values, we might want to clip, 
        # but for noiseless smooth functions, extrapolation is usually 
        # reasonable over small distances.
        pred = np.nan_to_num(pred, nan=0.0, posinf=1e10, neginf=-1e10)
        
        return pred

    return predict
