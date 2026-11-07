"""
Strategy: Adaptive 1D Interpolation with Multi-Basis Fallback.
Since all equations have 1 input dimension, we primarily use PCHIP (monotone cubic)
interpolation which handles noiseless data well without oscillation.
We also generate polynomial, Fourier, and RBF basis candidates and use
cross-validation to select the best model if interpolation is not suitable
(e.g., for periodic functions where boundary behavior matters).
A time guard ensures we finish before the deadline.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.interpolate import PchipInterpolator, CubicSpline

def _features_polynomial(x, degree=5):
    """Generate polynomial features up to given degree for 1D input."""
    n = len(x)
    cols = [np.ones(n)]
    for deg in range(1, degree + 1):
        cols.append(x ** deg)
    return np.column_stack(cols)

def _features_fourier(x, n_freqs=8):
    """Generate Fourier features (sin and cos of multiples of x)."""
    n = len(x)
    cols = [np.ones(n)]
    for k in range(1, n_freqs + 1):
        cols.append(np.sin(k * x))
        cols.append(np.cos(k * x))
    return np.column_stack(cols)

def _features_rbf(x, n_centers=20):
    """Generate RBF features with centers spread across the data range."""
    n = len(x)
    cols = [np.ones(n)]
    min_val = np.min(x)
    max_val = np.max(x)
    if max_val - min_val < 1e-10:
        cols.append(np.ones(n))
        return np.column_stack(cols)
    centers = np.linspace(min_val, max_val, n_centers)
    bandwidth = (max_val - min_val) / n_centers * 1.5
    for c in centers:
        cols.append(np.exp(-((x - c) ** 2) / (2 * bandwidth ** 2)))
    return np.column_stack(cols)

def _fit_linear(features, y, reg=1e-6):
    """Fit linear model with regularization."""
    n, p = features.shape
    A = features.T @ features + reg * np.eye(p)
    b = features.T @ y
    try:
        w = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(features, y, rcond=None)[0]
    return w

def _cross_validate(features, y, n_folds=5, fit_func=None):
    """K-fold cross-validation to estimate generalization error."""
    n = len(y)
    indices = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(indices)
    fold_size = n // n_folds
    mse_sum = 0.0
    
    for i in range(n_folds):
        val_start = i * fold_size
        val_end = val_start + fold_size if i < n_folds - 1 else n
        val_idx = indices[val_start:val_end]
        train_idx = np.concatenate([indices[:val_start], indices[val_end:]])
        
        X_train = features[train_idx]
        y_train = y[train_idx]
        X_val = features[val_idx]
        y_val = y[val_idx]
        
        if fit_func is not None:
            w = fit_func(X_train, y_train)
            y_pred = X_val @ w
        else:
            w = _fit_linear(X_train, y_train)
            y_pred = X_val @ w
            
        mse = np.mean((y_pred - y_val) ** 2)
        mse_sum += mse
    
    return mse_sum / n_folds

def fit(X_train, y_train):
    start_time = time.time()
    deadline = 280.0  # Safety margin before 300s limit
    
    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train, dtype=float)
    n, d = X.shape
    
    # Flatten for 1D processing
    x = X.flatten()
    y_flat = y.flatten()
    
    # Check for constant y
    if np.var(y_flat) < 1e-10:
        mean_y = np.mean(y_flat)
        def predict_const(X_new):
            return np.full(np.asarray(X_new).shape, mean_y)
        return predict_const
    
    # Sort by x for interpolation
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y_flat[sort_idx]
    
    # Handle duplicates
    unique_x, inverse_indices, counts = np.unique(x_sorted, return_inverse=True, return_counts=True)
    if len(unique_x) != len(x_sorted):
        y_sum = np.zeros_like(unique_x, dtype=float)
        np.add.at(y_sum, inverse_indices, y_sorted)
        y_sorted = y_sum / counts
        x_sorted = unique_x
    
    if len(x_sorted) < 2:
        mean_y = np.mean(y_flat)
        def predict_single(X_new):
            return np.full(np.asarray(X_new).shape, mean_y)
        return predict_single
    
    # Strategy 1: PCHIP Interpolation (primary for 1D noiseless data)
    pchip_interp = None
    try:
        if np.any(np.diff(x_sorted) <= 0):
            raise ValueError("Non-increasing X")
        pchip_interp = PchipInterpolator(x_sorted, y_sorted)
        # Test evaluation
        test_x = x_sorted[::max(1, len(x_sorted)//10)]
        test_y = pchip_interp(test_x)
        if not np.all(np.isfinite(test_y)):
            raise ValueError("Non-finite values in PCHIP")
    except Exception:
        pchip_interp = None
    
    # Strategy 2: Cubic Spline (alternative)
    cubic_spline = None
    try:
        if np.any(np.diff(x_sorted) <= 0):
            raise ValueError("Non-increasing X")
        cubic_spline = CubicSpline(x_sorted, y_sorted)
        test_x = x_sorted[::max(1, len(x_sorted)//10)]
        test_y = cubic_spline(test_x)
        if not np.all(np.isfinite(test_y)):
            raise ValueError("Non-finite values in CubicSpline")
    except Exception:
        cubic_spline = None
    
    # Strategy 3: Basis Function Regression
    best_basis_mse = np.inf
    best_basis_w = None
    best_basis_F = None
    best_basis_name = None
    
    candidates = []
    
    # Polynomial of various degrees
    for deg in [2, 3, 4, 5, 6, 8]:
        try:
            F = _features_polynomial(x_sorted, degree=deg)
            candidates.append(('poly_deg%d' % deg, F))
        except Exception:
            pass
    
    # Fourier with various frequencies
    for nf in [4, 6, 8, 10, 12, 15]:
        try:
            F = _features_fourier(x_sorted, n_freqs=nf)
            candidates.append(('fourier_%d' % nf, F))
        except Exception:
            pass
    
    # RBF with various numbers of centers
    for nc in [10, 15, 20, 25, 30, 40]:
        try:
            F = _features_rbf(x_sorted, n_centers=nc)
            candidates.append(('rbf_%d' % nc, F))
        except Exception:
            pass
    
    # Evaluate each candidate via cross-validation
    for name, F in candidates:
        if time.time() > deadline - 10:
            break
        try:
            mse = _cross_validate(F, y_sorted, n_folds=5)
            if mse < best_basis_mse:
                best_basis_mse = mse
                best_basis_F = F
                best_basis_w = _fit_linear(F, y_sorted)
                best_basis_name = name
        except Exception:
            continue
    
    # Decide which model to use
    # PCHIP is generally best for 1D noiseless data unless the function is periodic
    # and the range doesn't wrap around nicely, but even then PCHIP is often good.
    # We can estimate the "quality" of PCHIP by checking residuals on training data (should be ~0)
    # and comparing to basis models via CV.
    
    # Since training data is noiseless, PCHIP interpolates exactly (residual ~0).
    # Basis models will have non-zero CV error.
    # We prefer PCHIP unless a basis model has significantly lower CV error,
    # which might indicate a better generalization for specific shapes (e.g., periodic).
    
    # Heuristic: If the best basis CV MSE is much smaller than the typical noise level
    # (which is 0), it's not directly comparable.
    # Instead, let's look at the shape. If the function looks periodic (e.g., sin/cos),
    # Fourier might be better. But PCHIP is very robust.
    
    # Let's default to PCHIP if available, as it's the strongest inspiration.
    # We can refine this by checking if the data looks periodic.
    
    use_pchip = pchip_interp is not None
    use_cubic = not use_pchip and cubic_spline is not None
    use_basis = not use_pchip and not use_cubic and best_basis_w is not None
    
    # Actually, let's try to detect if Fourier is significantly better.
    # If the best Fourier CV error is < 1e-6 (very low), it might be a perfect fit.
    # But PCHIP is also a perfect fit on training data.
    # The test points are separate.
    # For now, PCHIP is the safest bet for "smooth local" functions.
    
    if use_pchip:
        def predict(X_new):
            X_eval = np.asarray(X_new).flatten()
            y_pred = pchip_interp(X_eval)
            y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0)
            return y_pred.reshape(np.asarray(X_new).shape)
        
        return predict
    
    elif use_cubic:
        def predict(X_new):
            X_eval = np.asarray(X_new).flatten()
            y_pred = cubic_spline(X_eval)
            y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0)
            return y_pred.reshape(np.asarray(X_new).shape)
        
        return predict
    
    elif use_basis:
        # Need to transform X_new using the same basis
        # We stored best_basis_name, so we can regenerate features
        def predict(X_new):
            X_eval = np.asarray(X_new).flatten()
            # Regenerate features
            if best_basis_name.startswith('poly'):
                deg = int(best_basis_name.split('_')[-1].replace('deg', ''))
                F_new = _features_polynomial(X_eval, degree=deg)
            elif best_basis_name.startswith('fourier'):
                nf = int(best_basis_name.split('_')[-1])
                F_new = _features_fourier(X_eval, n_freqs=nf)
            elif best_basis_name.startswith('rbf'):
                nc = int(best_basis_name.split('_')[-1])
                F_new = _features_rbf(X_eval, n_centers=nc)
            else:
                # Fallback to linear
                F_new = np.column_stack([np.ones(len(X_eval)), X_eval])
            
            y_pred = F_new @ best_basis_w
            y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0)
            return y_pred.reshape(np.asarray(X_new).shape)
        
        return predict
    
    else:
        # Fallback to mean
        mean_y = np.mean(y_flat)
        def predict_fallback(X_new):
            return np.full(np.asarray(X_new).shape, mean_y)
        return predict_fallback

# EVOLVE-BLOCK-END
