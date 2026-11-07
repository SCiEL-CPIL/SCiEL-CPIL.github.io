"""Structural change: Robust 1D Regression with Sorted Data and Model Selection (Poly vs Spline)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.interpolate import CubicSpline

def _poly_features(x, degree):
    """
    Generate polynomial features for 1D input x.
    Returns array of shape (n, degree+1) with columns [1, x, x^2, ..., x^degree]
    """
    x = np.asarray(x).reshape(-1, 1)
    features = np.ones((x.shape[0], degree + 1))
    for i in range(1, degree + 1):
        features[:, i] = x[:, 0] ** i
    return features

def _ridge_fit(x, y, alpha=1.0):
    """
    Fit Ridge regression y = Xw + b (handled by including bias in X).
    Returns weights w.
    """
    # x is assumed to be the feature matrix including bias term
    # Solve (X^T X + alpha * I) w = X^T y
    # Note: Usually alpha is not applied to the bias term, but for simplicity and stability 
    # with small data, applying it to all or just handling via centering is common.
    # Standard Ridge: (X^T X + alpha I) w = X^T y
    # To avoid regularizing bias, we can center X and y, fit, then adjust.
    # But for 1D poly regression with alpha=1.0 and n=200, simple ridge on augmented matrix is often stable enough.
    # Let's do proper ridge without regularizing bias.
    
    X = x
    y = y
    n, p = X.shape
    
    # Center X and y
    X_mean = np.mean(X, axis=0)
    y_mean = np.mean(y)
    X_centered = X - X_mean
    y_centered = y - y_mean
    
    # Compute X^T X and X^T y
    XtX = X_centered.T @ X_centered
    Xty = X_centered.T @ y_centered
    
    # Add regularization to diagonal (except bias term if we kept it, but we centered so no explicit bias row needed in this formulation? 
    # Actually, centering removes the need for an explicit bias column in the regularized part, but we need to recover the intercept.
    # Let's stick to the augmented matrix approach which is easier to map to the previous code structure, 
    # but modify the regularization to exclude the first column (bias).
    
    # Revert to augmented matrix approach for clarity and consistency with previous logic
    # X_aug = [1, x, x^2, ...]
    # (X_aug^T X_aug + alpha * diag(0, 1, 1, ...)) w = X_aug^T y
    
    A = X.T @ X
    b = X.T @ y
    
    # Create regularization matrix
    reg = alpha * np.eye(p)
    reg[0, 0] = 0.0  # Do not regularize bias
    
    A_reg = A + reg
    
    try:
        w = np.linalg.solve(A_reg, b)
    except np.linalg.LinAlgError:
        # Fallback to least squares if singular (shouldn't happen with ridge)
        w = np.linalg.lstsq(X, y, rcond=None)[0]
        
    return w

def _ridge_predict(x, w):
    """
    Predict using Ridge weights w.
    """
    return x @ w

_START_TIME = None
TIME_LIMIT = 300.0
SAFETY_MARGIN = 15.0

def _time_left():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
        return TIME_LIMIT - SAFETY_MARGIN
    return TIME_LIMIT - SAFETY_MARGIN - (time.time() - _START_TIME)

def _cv_score(x, y, model_type, param, n_folds=5):
    """
    Calculate CV MSE for a given model type and parameter.
    model_type: 'poly' or 'spline'
    param: degree for poly, ignored for spline (or alpha for ridge)
    """
    n = len(x)
    indices = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(indices)
    
    fold_size = n // n_folds
    fold_mses = []
    
    for i in range(n_folds):
        if _time_left() < 2:
            return np.inf
            
        val_idx = indices[i*fold_size : (i+1)*fold_size]
        train_idx = np.setdiff1d(indices, val_idx)
        
        x_tr = x[train_idx]
        y_tr = y[train_idx]
        x_val = x[val_idx]
        y_val = y[val_idx]
        
        try:
            if model_type == 'spline':
                # Spline requires sorted data
                sort_idx = np.argsort(x_tr)
                x_tr_sorted = x_tr[sort_idx]
                y_tr_sorted = y_tr[sort_idx]
                
                # Check for duplicates
                if np.any(np.diff(x_tr_sorted) == 0):
                    # If duplicates, spline fails or is ill-defined. 
                    # For noiseless data, duplicates in x should have same y.
                    # We can average them or just fail this fold.
                    # Let's try to handle by removing duplicates
                    unique_x, unique_indices = np.unique(x_tr_sorted, return_index=True)
                    if len(unique_x) < 4:
                        fold_mses.append(np.inf)
                        continue
                    y_tr_sorted = y_tr_sorted[unique_indices]
                    x_tr_sorted = unique_x
                    
                cs = CubicSpline(x_tr_sorted, y_tr_sorted)
                pred = cs(x_val)
            elif model_type == 'poly':
                deg = param
                # Use Ridge for stability
                # Generate features: [1, x, x^2, ..., x^deg]
                X_tr_poly = _poly_features(x_tr, deg)
                X_val_poly = _poly_features(x_val, deg)
                
                # Ridge regression
                # alpha=1.0 is a reasonable default
                w = _ridge_fit(X_tr_poly, y_tr, alpha=1.0)
                pred = _ridge_predict(X_val_poly, w)
            else:
                return np.inf
                
            mse = np.mean((pred - y_val)**2)
            fold_mses.append(mse)
        except Exception:
            fold_mses.append(np.inf)
            
    if not fold_mses:
        return np.inf
    return np.mean(fold_mses)

def fit(X_train, y_train):
    global _START_TIME
    _START_TIME = time.time()
    
    x = X_train[:, 0].copy()
    y = y_train.copy()
    n = len(x)
    
    # Sort data for Spline compatibility and general stability
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]
    
    # Handle constant y
    if np.std(y) < 1e-10:
        def predict(X):
            return np.full(X.shape[0], np.mean(y))
        return predict

    # Candidate models
    # 1. Spline (1 model)
    # 2. Polynomials (deg 1 to 8)
    
    best_mse = np.inf
    best_model = None
    best_param = None
    
    # Test Spline
    if _time_left() > 5:
        mse_spline = _cv_score(x_sorted, y_sorted, 'spline', None)
        if mse_spline < best_mse:
            best_mse = mse_spline
            best_model = 'spline'
            best_param = None

    # Test Polynomials
    for deg in range(1, 9):
        if _time_left() < 5:
            break
        mse_poly = _cv_score(x_sorted, y_sorted, 'poly', deg)
        if mse_poly < best_mse:
            best_mse = mse_poly
            best_model = 'poly'
            best_param = deg

    # If CV failed for all (unlikely), fallback to linear
    if best_model is None:
        best_model = 'poly'
        best_param = 1

    # Refit on all data
    if best_model == 'spline':
        # Ensure unique x for spline
        unique_x, unique_indices = np.unique(x_sorted, return_index=True)
        if len(unique_x) < 4:
             # Fallback to linear if too few points
             best_model = 'poly'
             best_param = 1
        else:
             y_unique = y_sorted[unique_indices]
             try:
                 cs = CubicSpline(unique_x, y_unique)
                 def predict(X):
                     X_test = X[:, 0]
                     pred = cs(X_test)
                     return np.asarray(pred, dtype=float)
             except Exception:
                 # Fallback to linear
                 best_model = 'poly'
                 best_param = 1
                 
    if best_model == 'poly':
        deg = best_param
        # Generate features: [1, x, x^2, ..., x^deg]
        X_poly = _poly_features(x_sorted, deg)
        
        w = _ridge_fit(X_poly, y_sorted, alpha=1.0)
        
        def predict(X):
            X_test = X[:, 0].reshape(-1, 1)
            X_test_poly = _poly_features(X_test, deg)
            pred = _ridge_predict(X_test_poly, w)
            return np.asarray(pred, dtype=float)

    # Safety wrapper
    def safe_predict(X):
        try:
            res = predict(X)
            res = np.asarray(res, dtype=float)
            # Replace non-finite with mean
            mean_y = np.mean(y)
            res = np.where(np.isfinite(res), res, mean_y)
            return res
        except Exception:
            return np.full(X.shape[0], np.mean(y))
            
    return safe_predict
# EVOLVE-BLOCK-END
