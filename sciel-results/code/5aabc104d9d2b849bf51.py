import numpy as np
from scipy.interpolate import PchipInterpolator, CubicSpline
import time

def fit(X_train, y_train):
    start_time = time.time()
    deadline = 280.0  # Safety margin
    
    X = np.asarray(X_train, dtype=float).flatten()
    y = np.asarray(y_train, dtype=float).flatten()
    
    # Sort by X
    sort_idx = np.argsort(X)
    X_sorted = X[sort_idx]
    y_sorted = y[sort_idx]
    
    # Handle duplicates
    unique_X, inverse_indices, counts = np.unique(X_sorted, return_inverse=True, return_counts=True)
    if len(unique_X) != len(X_sorted):
        y_sum = np.zeros_like(unique_X, dtype=float)
        np.add.at(y_sum, inverse_indices, y_sorted)
        y_sorted = y_sum / counts
        X_sorted = unique_X
        
    if len(X_sorted) < 2:
        mean_y = np.mean(y)
        return lambda X: np.full_like(X.flatten(), mean_y).reshape(X.shape)

    # Strategy 1: PCHIP Interpolator (Robust, no overshoot)
    pchip_interp = None
    pchip_error = np.inf
    try:
        pchip_interp = PchipInterpolator(X_sorted, y_sorted)
        y_pred_pchip = pchip_interp(X_sorted)
        pchip_error = np.sum((y_pred_pchip - y_sorted)**2)
    except Exception:
        pchip_interp = None

    # Strategy 2: Polynomial Fit (Try degrees 1 to 20)
    best_poly_deg = -1
    best_poly_coef = None
    best_poly_error = np.inf
    
    # Normalize X for better numerical stability in polyfit
    X_mean = np.mean(X_sorted)
    X_std = np.std(X_sorted)
    if X_std == 0:
        X_norm = X_sorted
    else:
        X_norm = (X_sorted - X_mean) / X_std

    for deg in range(1, 21):
        try:
            # Use np.polyfit on normalized X
            coef = np.polyfit(X_norm, y_sorted, deg)
            # Evaluate
            y_pred_poly = np.polyval(coef, X_norm)
            error = np.sum((y_pred_poly - y_sorted)**2)
            
            if error < best_poly_error:
                best_poly_error = error
                best_poly_coef = coef
                best_poly_deg = deg
        except Exception:
            continue
        
        # If error is very small, we can stop early
        if best_poly_error < 1e-10:
            break

    # Select the best model
    if pchip_interp is not None and best_poly_deg == -1:
        # Only PCHIP available
        def predict(X):
            X_eval = np.asarray(X, dtype=float).flatten()
            y_pred = pchip_interp(X_eval)
            return y_pred.reshape(np.asarray(X).shape)
        return predict
        
    elif pchip_interp is None and best_poly_deg != -1:
        # Only Poly available
        def predict(X):
            X_eval = np.asarray(X, dtype=float).flatten()
            X_eval_norm = (X_eval - X_mean) / X_std
            y_pred = np.polyval(best_poly_coef, X_eval_norm)
            return y_pred.reshape(np.asarray(X).shape)
        return predict
        
    else:
        # Both available, pick lower training error
        if pchip_error < best_poly_error:
            def predict(X):
                X_eval = np.asarray(X, dtype=float).flatten()
                y_pred = pchip_interp(X_eval)
                return y_pred.reshape(np.asarray(X).shape)
            return predict
        else:
            def predict(X):
                X_eval = np.asarray(X, dtype=float).flatten()
                X_eval_norm = (X_eval - X_mean) / X_std
                y_pred = np.polyval(best_poly_coef, X_eval_norm)
                return y_pred.reshape(np.asarray(X).shape)
            return predict
