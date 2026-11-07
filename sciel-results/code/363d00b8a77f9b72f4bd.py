# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.optimize import minimize_scalar

DEADLINE = 300.0  # seconds total
SAFETY_MARGIN = 10.0  # seconds to leave before kill

def _make_spline_predictor(x_sorted, y_sorted, k=3, s=0.0):
    """Create a spline-based predictor. Returns predict function."""
    try:
        # Use UnivariateSpline with smoothing parameter s
        # s=0 gives interpolation (passing through all points)
        spl = UnivariateSpline(x_sorted, y_sorted, k=k, s=s)
        def predict(X):
            return spl(X[:, 0])
        return predict
    except Exception:
        # Fallback to linear interpolation
        f = interp1d(x_sorted, y_sorted, kind='linear', fill_value='extrapolate')
        def predict(X):
            return f(X[:, 0])
        return predict

def _fit_spline(X_train, y_train, k=3, s=0.0):
    """Fit a spline to 1D data and return a predictor."""
    x = X_train[:, 0]
    y = y_train
    
    # Sort by x
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]
    
    # Remove duplicate x values (keep first occurrence)
    unique_mask = np.concatenate([True, np.diff(x_sorted) > 1e-12])
    x_unique = x_sorted[unique_mask]
    y_unique = y_sorted[unique_mask]
    
    if len(x_unique) < k + 1:
        # Not enough points for spline, use linear interpolation
        f = interp1d(x_unique, y_unique, kind='linear', fill_value='extrapolate')
        def predict(X):
            return f(X[:, 0])
        return predict
    
    try:
        spl = UnivariateSpline(x_unique, y_unique, k=k, s=s)
        def predict(X):
            return spl(X[:, 0])
        return predict
    except Exception:
        f = interp1d(x_unique, y_unique, kind='linear', fill_value='extrapolate')
        def predict(X):
            return f(X[:, 0])
        return predict

def _fit_polynomial(X_train, y_train, degree=10):
    """Fit a polynomial and return a predictor."""
    x = X_train[:, 0]
    y = y_train
    
    # Fit polynomial
    coeffs = np.polyfit(x, y, degree)
    
    def predict(X):
        return np.polyval(coeffs, X[:, 0])
    return predict

def _fit_piecewise_linear(X_train, y_train, n_bins=20):
    """Fit piecewise linear function using binning."""
    x = X_train[:, 0]
    y = y_train
    
    x_min, x_max = x.min(), x.max()
    if x_max - x_min < 1e-12:
        return lambda X: np.full(len(X), y.mean())
    
    # Create bins
    edges = np.linspace(x_min, x_max, n_bins + 1)
    bin_indices = np.digitize(x, edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    # Compute mean y per bin
    bin_means = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_means[i] = y[mask].mean()
            bin_counts[i] = mask.sum()
    
    # Fill empty bins with neighboring values
    for i in range(n_bins):
        if bin_counts[i] == 0:
            # Find nearest non-empty bin
            for d in range(1, n_bins):
                if i - d >= 0 and bin_counts[i - d] > 0:
                    bin_means[i] = bin_means[i - d]
                    break
                if i + d < n_bins and bin_counts[i + d] > 0:
                    bin_means[i] = bin_means[i + d]
                    break
    
    # Create interpolation function
    bin_centers = (edges[:-1] + edges[1:]) / 2
    
    def predict(X):
        xi = X[:, 0]
        # Clip to range
        xi_clipped = np.clip(xi, x_min, x_max)
        # Linear interpolation between bin centers
        return np.interp(xi_clipped, bin_centers, bin_means)
    
    return predict

def _cross_validate_spline(X_train, y_train, k=3, s_values=None):
    """Cross-validate to find best smoothing parameter."""
    if s_values is None:
        s_values = [0.0, 1e-6, 1e-4, 1e-2, 0.1, 1.0, 10.0]
    
    x = X_train[:, 0]
    y = y_train
    n = len(x)
    
    # Simple hold-out validation
    np.random.seed(42)
    indices = np.random.permutation(n)
    n_val = max(10, n // 5)
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]
    
    x_train_cv = x[train_idx]
    y_train_cv = y[train_idx]
    x_val = x[val_idx]
    y_val = y[val_idx]
    
    order = np.argsort(x_train_cv)
    x_sorted = x_train_cv[order]
    y_sorted = y_train_cv[order]
    
    unique_mask = np.concatenate([True, np.diff(x_sorted) > 1e-12])
    x_unique = x_sorted[unique_mask]
    y_unique = y_sorted[unique_mask]
    
    best_s = 0.0
    best_mse = np.inf
    
    for s in s_values:
        try:
            spl = UnivariateSpline(x_unique, y_unique, k=k, s=s)
            pred = spl(x_val)
            mse = np.mean((pred - y_val) ** 2)
            if mse < best_mse:
                best_mse = mse
                best_s = s
        except Exception:
            continue
    
    return best_s

def fit(X_train, y_train):
    start_time = time.time()
    
    # Check if 1D
    if X_train.shape[1] == 1:
        x = X_train[:, 0]
        y = y_train
    else:
        # For multi-dim, use polynomial fit as fallback
        # But task says all are 1D, so this shouldn't happen
        d = X_train.shape[1]
        # Use a simple linear + quadratic model
        def features(X):
            cols = [np.ones(len(X))]
            for j in range(d):
                cols.append(X[:, j])
            for i in range(d):
                for j in range(i, d):
                    cols.append(X[:, i] * X[:, j])
            return np.column_stack(cols)
        F = features(X_train)
        w, *_ = np.linalg.lstsq(F, y_train, rcond=None)
        return lambda X: features(X) @ w
    
    # 1D case: try multiple approaches and pick best
    n = len(x)
    
    # Approach 1: Interpolating spline (s=0)
    try:
        pred_spline0 = _fit_spline(X_train, y_train, k=3, s=0.0)
        # Evaluate on training data
        train_pred = pred_spline0(X_train)
        mse_spline0 = np.mean((train_pred - y_train) ** 2)
    except Exception:
        mse_spline0 = np.inf
    
    # Approach 2: Polynomial fit (degree 10)
    try:
        pred_poly = _fit_polynomial(X_train, y_train, degree=10)
        train_pred = pred_poly(X_train)
        mse_poly = np.mean((train_pred - y_train) ** 2)
    except Exception:
        mse_poly = np.inf
    
    # Approach 3: Piecewise linear with more bins
    try:
        pred_pl = _fit_piecewise_linear(X_train, y_train, n_bins=50)
        train_pred = pred_pl(X_train)
        mse_pl = np.mean((train_pred - y_train) ** 2)
    except Exception:
        mse_pl = np.inf
    
    # Approach 4: Spline with small smoothing
    try:
        pred_spline_small = _fit_spline(X_train, y_train, k=3, s=1e-8)
        train_pred = pred_spline_small(X_train)
        mse_spline_small = np.mean((train_pred - y_train) ** 2)
    except Exception:
        mse_spline_small = np.inf
    
    # Pick the best based on training MSE (since data is noiseless, 
    # the best interpolator should be best)
    candidates = [
        (mse_spline0, pred_spline0),
        (mse_poly, pred_poly),
        (mse_pl, pred_pl),
        (mse_spline_small, pred_spline_small),
    ]
    
    # Filter out infinite MSEs
    valid_candidates = [(m, p) for m, p in candidates if np.isfinite(m)]
    
    if not valid_candidates:
        # Fallback: linear interpolation
        order = np.argsort(x)
        x_sorted = x[order]
        y_sorted = y_train[order]
        f = interp1d(x_sorted, y_sorted, kind='linear', fill_value='extrapolate')
        return lambda X: f(X[:, 0])
    
    # Sort by MSE and pick best
    valid_candidates.sort(key=lambda t: t[0])
    best_predictor = valid_candidates[0][1]
    
    return best_predictor
# EVOLVE-BLOCK-END
