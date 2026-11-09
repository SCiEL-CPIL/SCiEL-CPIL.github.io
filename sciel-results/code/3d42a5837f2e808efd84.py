# EVOLVE-BLOCK-START
import numpy as np
from scipy.interpolate import PchipInterpolator, interp1d
import warnings

def fit(X_train, y_train):
    """
    Fit a regression model using a hybrid approach:
    1. Attempt to fit low-degree polynomials (exact for polynomial ground truths).
    2. If no low-degree polynomial fits well, use PchipInterpolator (monotonicity-preserving cubic spline).
    3. Fallback to linear interpolation.
    """
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    n_samples, n_features = X.shape
    
    if n_features > 1:
        # Fallback for multi-feature (should not happen per task description)
        from numpy.polynomial import polynomial as P
        X_1d = X[:, 0]
        powers = np.column_stack([X_1d**i for i in range(5)])
        w, *_ = np.linalg.lstsq(powers, y, rcond=None)
        def predict_poly(X_test):
            X_test = np.asarray(X_test, dtype=np.float64)
            if X_test.ndim == 1:
                X_test = X_test.reshape(-1, 1)
            X_1d = X_test[:, 0]
            powers_test = np.column_stack([X_1d**i for i in range(5)])
            return powers_test @ w
        return predict_poly

    x = X[:, 0]
    
    # Sort by x
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]
    
    # Handle duplicates
    unique_x, unique_indices = np.unique(x_sorted, return_index=True)
    if len(unique_x) < len(x_sorted):
        y_unique = np.zeros(len(unique_x))
        for i, idx in enumerate(unique_indices):
            if i < len(unique_indices) - 1:
                next_idx = unique_indices[i+1]
                y_unique[i] = y_sorted[idx:next_idx].mean()
            else:
                y_unique[i] = y_sorted[idx:].mean()
        x_sorted = unique_x
        y_sorted = y_unique
        
    n_unique = len(x_sorted)
    
    # 1. Try polynomial fits
    # If the function is a polynomial of degree d, this will find it exactly.
    for deg in range(1, 10):
        if n_unique <= deg:
            continue
        try:
            coeffs = np.polyfit(x_sorted, y_sorted, deg)
            y_pred = np.polyval(coeffs, x_sorted)
            residuals = y_sorted - y_pred
            # Check relative error
            norm_y = np.linalg.norm(y_sorted)
            if norm_y > 1e-10:
                rel_err = np.linalg.norm(residuals) / norm_y
            else:
                rel_err = np.linalg.norm(residuals)
            
            if rel_err < 1e-8:
                # Found a good polynomial fit
                def predict_poly(X_test, coeffs=coeffs):
                    X_test = np.asarray(X_test, dtype=np.float64)
                    if X_test.ndim == 1:
                        X_test = X_test.reshape(-1, 1)
                    x_test = X_test[:, 0]
                    return np.polyval(coeffs, x_test)
                return predict_poly
        except Exception:
            continue
            
    # 2. Use PchipInterpolator for general smooth functions
    # Pchip is more robust than UnivariateSpline for non-uniform data and avoids oscillations
    try:
        interpolator = PchipInterpolator(x_sorted, y_sorted)
        def predict_pchip(X_test, interpolator=interpolator):
            X_test = np.asarray(X_test, dtype=np.float64)
            if X_test.ndim == 1:
                X_test = X_test.reshape(-1, 1)
            x_test = X_test[:, 0]
            return interpolator(x_test)
        return predict_pchip
    except Exception:
        pass
        
    # 3. Fallback to linear interpolation
    try:
        interpolator = interp1d(x_sorted, y_sorted, kind='linear', fill_value='extrapolate')
        def predict_linear(X_test, interpolator=interpolator):
            X_test = np.asarray(X_test, dtype=np.float64)
            if X_test.ndim == 1:
                X_test = X_test.reshape(-1, 1)
            x_test = X_test[:, 0]
            return interpolator(x_test)
        return predict_linear
    except Exception:
        # Last resort: constant prediction (mean)
        mean_y = np.mean(y_sorted)
        def predict_const(X_test, mean_y=mean_y):
            X_test = np.asarray(X_test, dtype=np.float64)
            if X_test.ndim == 1:
                X_test = X_test.reshape(-1, 1)
            return np.full(X_test.shape[0], mean_y)
        return predict_const
# EVOLVE-BLOCK-END
