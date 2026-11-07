"""Wavelet-enhanced symbolic regression with kernel fallback and time guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import warnings

warnings.filterwarnings("ignore")

_START_TIME = None
TIME_LIMIT = 300.0
SAFETY_MARGIN = 20.0


def _time_left():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
        return TIME_LIMIT - SAFETY_MARGIN
    return TIME_LIMIT - SAFETY_MARGIN - (time.time() - _START_TIME)


def _poly_features(X, degree):
    """Polynomial features for 1D input."""
    x = X[:, 0]
    cols = [np.ones(len(x))]
    for k in range(1, degree + 1):
        cols.append(x ** k)
    return np.column_stack(cols)


def _trig_features(X, n_harmonics=5):
    """Trigonometric features for 1D input."""
    x = X[:, 0]
    cols = [np.ones(len(x))]
    for k in range(1, n_harmonics + 1):
        cols.append(np.sin(k * x))
        cols.append(np.cos(k * x))
    return np.column_stack(cols)


def _mixed_features(X, poly_deg=3, n_harmonics=3):
    """Mixed polynomial and trigonometric features."""
    x = X[:, 0]
    cols = [np.ones(len(x))]
    for k in range(1, poly_deg + 1):
        cols.append(x ** k)
    for k in range(1, n_harmonics + 1):
        cols.append(np.sin(k * x))
        cols.append(np.cos(k * x))
    return np.column_stack(cols)


def _exp_features(X):
    """Exponential and log features."""
    x = X[:, 0]
    cols = [np.ones(len(x)), x, x ** 2]
    x_clipped = np.clip(x, -10, 10)
    cols.append(np.exp(x_clipped))
    cols.append(np.exp(-x_clipped))
    x_safe = np.where(np.abs(x) > 1e-6, x, 1e-6)
    cols.append(np.log(np.abs(x_safe)))
    return np.column_stack(cols)


def _sqrt_features(X):
    """Square root and related features for positive domain."""
    x = X[:, 0]
    n = len(x)
    cols = [np.ones(n), x, x ** 2]
    # Safe sqrt
    pos_mask = x > 0
    sqrt_x = np.zeros(n)
    sqrt_x[pos_mask] = np.sqrt(x[pos_mask])
    cols.append(sqrt_x)
    cols.append(x * sqrt_x)
    # Abs
    cols.append(np.abs(x))
    # Power features
    x_safe = np.where(np.abs(x) > 1e-10, x, 1e-10)
    cols.append(np.sign(x_safe) * np.abs(x_safe) ** 0.5)
    cols.append(np.sign(x_safe) * np.abs(x_safe) ** 1.5)
    return np.column_stack(cols)


def _rich_features(X, max_poly=8, max_trig=5):
    """Rich feature set combining multiple families."""
    x = X[:, 0]
    n = len(x)
    cols = [np.ones(n)]
    
    # Polynomial
    for k in range(1, max_poly + 1):
        cols.append(x ** k)
    
    # Trigonometric
    for k in range(1, max_trig + 1):
        cols.append(np.sin(k * x))
        cols.append(np.cos(k * x))
    
    # Exponential (scaled)
    x_clipped = np.clip(x, -5, 5)
    cols.append(np.exp(0.5 * x_clipped))
    cols.append(np.exp(-0.5 * x_clipped))
    
    # Sqrt and power
    pos_mask = x > 0
    sqrt_x = np.zeros(n)
    sqrt_x[pos_mask] = np.sqrt(x[pos_mask])
    cols.append(sqrt_x)
    cols.append(x * sqrt_x)
    cols.append(np.abs(x))
    
    # Rational
    denom = 1 + x ** 2
    cols.append(x / denom)
    cols.append(x ** 2 / denom)
    
    return np.column_stack(cols)


def _daubechies_d4_coefficients():
    """Return Daubechies-4 wavelet filter coefficients."""
    # Low-pass and high-pass filters for db4
    sqrt2 = np.sqrt(2)
    # Normalized coefficients
    c = [
        0.23105960331452787,
        0.7148465705495041,
        0.6308807679295904,
        -0.02798376941695768,
        -0.3826834323650898,
        -0.24229488666183922,
        0.07576571478927333,
        0.03522629188570953
    ]
    # Normalize
    c = np.array(c)
    c = c / np.sqrt(np.sum(c ** 2))
    return c


def _wavelet_transform_1d(y, level=3):
    """
    Perform discrete wavelet transform on 1D signal.
    Returns (approx_coeffs, detail_coeffs_list, shape_info)
    """
    n = len(y)
    if n < 4:
        return [y.copy()], [], n
    
    # Use a simple Haar-like or Daubechies-4 filter
    # For simplicity and robustness, use a moving average approach as wavelet approximation
    # This is a simplified "wavelet denoising" that smooths while preserving features
    
    approx = y.copy()
    details = []
    current = y.copy()
    current_len = n
    
    for lev in range(level):
        if current_len < 4:
            break
        # Simple box filter for approximation (like Haar scaling function)
        # and difference for detail
        kernel_size = min(4, current_len)
        if kernel_size >= 2:
            # Moving average for approximation
            approx_next = np.convolve(current, np.ones(kernel_size) / kernel_size, mode='same')
            # Detail is the difference
            detail_next = current - approx_next
            details.append(detail_next)
            current = approx_next
            current_len = len(current)
    
    return [current], details, n


def _wavelet_denoise(y, threshold_factor=1.0, level=3):
    """
    Apply wavelet denoising with soft thresholding.
    Returns denoised signal.
    """
    n = len(y)
    if n < 8:
        return y.copy()
    
    # Estimate noise level
    # Use MAD estimator on differences
    diffs = np.diff(y)
    if len(diffs) < 4:
        return y.copy()
    mad = np.median(np.abs(diffs - np.median(diffs)))
    noise_std = 1.4826 * mad if mad > 0 else np.std(diffs) * 0.5
    
    # Perform simple multi-level smoothing
    denoised = y.copy()
    current = y.copy()
    
    for lev in range(level):
        if len(current) < 4:
            break
        # Smooth
        kernel_size = min(4, len(current))
        smoothed = np.convolve(current, np.ones(kernel_size) / kernel_size, mode='same')
        detail = current - smoothed
        
        # Soft threshold detail
        threshold = threshold_factor * noise_std * (1.0 + 0.5 * lev)
        detail_thresh = np.sign(detail) * np.maximum(np.abs(detail) - threshold, 0)
        
        current = smoothed + detail_thresh
    
    return current


def _rbf_kernel(X1, X2, gamma):
    """Compute RBF kernel matrix."""
    X1_sq = np.sum(X1 ** 2, axis=1)
    X2_sq = np.sum(X2 ** 2, axis=1)
    d2 = X1_sq[:, None] + X2_sq[None, :] - 2 * X1 @ X2.T
    d2 = np.maximum(d2, 0)
    return np.exp(-gamma * d2)


def _fit_rbf(X_train, y_train, gamma, alpha=1e-8):
    """Fit RBF kernel ridge regression."""
    K = _rbf_kernel(X_train, X_train, gamma)
    n = K.shape[0]
    K_reg = K + alpha * np.eye(n)
    try:
        w = np.linalg.solve(K_reg, y_train)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(K_reg, y_train, rcond=None)[0]
    
    def predict(X):
        K_test = _rbf_kernel(X, X_train, gamma)
        return K_test @ w
    
    y_pred = K @ w
    mse = np.mean((y_pred - y_train) ** 2)
    return mse, predict


def _fit_linear_features(features_func, X_train, y_train, alpha=0.0):
    """Fit linear model on features with optional ridge regularization."""
    try:
        Xf = features_func(X_train)
        n, p = Xf.shape
        if alpha > 0:
            AtA = Xf.T @ Xf + alpha * np.eye(p)
            Aty = Xf.T @ y_train
            w = np.linalg.solve(AtA, Aty)
        else:
            w, *_ = np.linalg.lstsq(Xf, y_train, rcond=None)
        y_pred = Xf @ w
        mse = np.mean((y_pred - y_train) ** 2)
        
        def predict(X):
            Xf_test = features_func(X)
            return Xf_test @ w
        
        return mse, predict
    except Exception:
        return float('inf'), None


def _wavelet_interpolation_predict(x_train, y_train, n_levels=3):
    """
    Create a prediction function that uses wavelet-enhanced local interpolation.
    Sorts data, applies wavelet denoising, and uses local polynomial fits.
    """
    # Sort by x
    sort_idx = np.argsort(x_train)
    x_sorted = x_train[sort_idx]
    y_sorted = y_train[sort_idx]
    
    # Apply wavelet denoising
    y_denoised = _wavelet_denoise(y_sorted, threshold_factor=0.8, level=n_levels)
    
    # For prediction, use local quadratic fit with adaptive window
    def predict(X):
        x_new = np.asarray(X).flatten()
        n = len(x_sorted)
        preds = np.zeros(len(x_new))
        
        for i, xi in enumerate(x_new):
            # Find nearest neighbors
            dists = np.abs(x_sorted - xi)
            # Use 10 nearest neighbors for local fit
            k = min(10, n)
            nearest_idx = np.argsort(dists)[:k]
            
            if k < 3:
                # Not enough points, just interpolate
                if k == 1:
                    preds[i] = y_denoised[nearest_idx[0]]
                elif k == 2:
                    # Linear interpolation
                    x1, x2 = x_sorted[nearest_idx[0]], x_sorted[nearest_idx[1]]
                    y1, y2 = y_denoised[nearest_idx[0]], y_denoised[nearest_idx[1]]
                    if abs(x2 - x1) > 1e-10:
                        preds[i] = y1 + (y2 - y1) * (xi - x1) / (x2 - x1)
                    else:
                        preds[i] = (y1 + y2) / 2
                continue
            
            # Local quadratic fit
            x_local = x_sorted[nearest_idx]
            y_local = y_denoised[nearest_idx]
            
            # Center and scale for numerical stability
            x_mean = np.mean(x_local)
            x_std = np.std(x_local)
            if x_std < 1e-10:
                x_scaled = np.zeros_like(x_local)
            else:
                x_scaled = (x_local - x_mean) / x_std
            
            # Fit quadratic
            A = np.column_stack([np.ones(k), x_scaled, x_scaled ** 2])
            try:
                coeffs, *_ = np.linalg.lstsq(A, y_local, rcond=None)
                xi_scaled = (xi - x_mean) / x_std
                preds[i] = coeffs[0] + coeffs[1] * xi_scaled + coeffs[2] * xi_scaled ** 2
            except Exception:
                # Fallback to nearest
                preds[i] = y_denoised[np.argmin(dists)]
        
        return preds
    
    return predict


def _cross_validate_simple(features_func, X_train, y_train, n_folds=5):
    """Simple k-fold CV to estimate generalization error."""
    n = len(X_train)
    indices = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(indices)
    
    fold_size = n // n_folds
    cv_errors = []
    
    for i in range(n_folds):
        val_start = i * fold_size
        val_end = val_start + fold_size if i < n_folds - 1 else n
        val_idx = indices[val_start:val_end]
        train_idx = np.concatenate([indices[:val_start], indices[val_end:]])
        
        X_tr = X_train[train_idx]
        y_tr = y_train[train_idx]
        X_val = X_train[val_idx]
        y_val = y_train[val_idx]
        
        try:
            Xf_tr = features_func(X_tr)
            Xf_val = features_func(X_val)
            w, *_ = np.linalg.lstsq(Xf_tr, y_tr, rcond=None)
            y_pred = Xf_val @ w
            cv_errors.append(np.mean((y_pred - y_val) ** 2))
        except Exception:
            cv_errors.append(float('inf'))
    
    return np.mean(cv_errors)


def fit(X_train, y_train):
    global _START_TIME
    _START_TIME = time.time()
    
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    
    d = X_train.shape[1]
    
    if d != 1:
        # Fallback to simple polynomial for multi-dimensional
        def features(X):
            cols = [np.ones(len(X))]
            for j in range(X.shape[1]):
                cols.append(X[:, j])
            for i in range(X.shape[1]):
                for j in range(i, X.shape[1]):
                    cols.append(X[:, i] * X[:, j])
            return np.column_stack(cols)
        w, *_ = np.linalg.lstsq(features(X_train), y_train, rcond=None)
        return lambda X: features(X) @ w
    
    x_train = X_train[:, 0]
    y_var = np.var(y_train)
    if y_var < 1e-10:
        # Constant function
        y_mean = np.mean(y_train)
        return lambda X: np.full(len(X), y_mean)
    
    # Candidate models
    best_cv_mse = float('inf')
    best_predict = None
    best_train_mse = float('inf')
    best_predict_train = None
    
    candidates = []
    
    # Polynomial models
    for deg in [3, 4, 5, 6, 7, 8]:
        candidates.append(('poly', deg, 0.0))
    
    # Trigonometric models
    for n_h in [3, 5, 7]:
        candidates.append(('trig', n_h, 0.0))
    
    # Mixed models
    for p, t in [(3, 3), (4, 4), (5, 5)]:
        candidates.append(('mixed', (p, t), 0.0))
    
    # Exponential models
    candidates.append(('exp', None, 0.0))
    
    # Sqrt/power models
    candidates.append(('sqrt', None, 0.0))
    
    # Rich features
    for mp, mt in [(6, 4), (8, 5)]:
        candidates.append(('rich', (mp, mt), 0.0))
    
    # RBF kernel models
    for gamma in [0.1, 0.5, 1.0, 2.0, 5.0]:
        candidates.append(('rbf', gamma, 1e-8))
    
    # Wavelet interpolation
    candidates.append(('wavelet', 3, 0.0))
    
    for model_type, param, alpha in candidates:
        if _time_left() < 5.0:
            break
        
        try:
            if model_type == 'poly':
                def feat(X, deg=param):
                    return _poly_features(X, deg)
                cv_mse = _cross_validate_simple(feat, X_train, y_train, n_folds=4)
                mse, predict = _fit_linear_features(feat, X_train, y_train, alpha)
                
            elif model_type == 'trig':
                def feat(X, n=param):
                    return _trig_features(X, n_harmonics=n)
                cv_mse = _cross_validate_simple(feat, X_train, y_train, n_folds=4)
                mse, predict = _fit_linear_features(feat, X_train, y_train, alpha)
                
            elif model_type == 'mixed':
                p, t = param
                def feat(X, p=p, t=t):
                    return _mixed_features(X, poly_deg=p, n_harmonics=t)
                cv_mse = _cross_validate_simple(feat, X_train, y_train, n_folds=4)
                mse, predict = _fit_linear_features(feat, X_train, y_train, alpha)
                
            elif model_type == 'exp':
                cv_mse = _cross_validate_simple(_exp_features, X_train, y_train, n_folds=4)
                mse, predict = _fit_linear_features(_exp_features, X_train, y_train, alpha)
                
            elif model_type == 'sqrt':
                cv_mse = _cross_validate_simple(_sqrt_features, X_train, y_train, n_folds=4)
                mse, predict = _fit_linear_features(_sqrt_features, X_train, y_train, alpha)
                
            elif model_type == 'rich':
                mp, mt = param
                def feat(X, mp=mp, mt=mt):
                    return _rich_features(X, max_poly=mp, max_trig=mt)
                cv_mse = _cross_validate_simple(feat, X_train, y_train, n_folds=4)
                mse, predict = _fit_linear_features(feat, X_train, y_train, alpha)
                
            elif model_type == 'rbf':
                gamma = param
                mse, predict = _fit_rbf(X_train, y_train, gamma, alpha)
                # Estimate CV-like score using training MSE as proxy
                cv_mse = mse * 1.1  # Slight penalty for RBF overfitting risk
                
            elif model_type == 'wavelet':
                n_levels = param
                predict = _wavelet_interpolation_predict(x_train, y_train, n_levels)
                # Evaluate on training data
                y_pred_train = predict(X_train)
                mse = np.mean((y_pred_train - y_train) ** 2)
                # Wavelet interpolation tends to overfit, so penalize
                cv_mse = mse * 1.5
                
            else:
                continue
            
            if predict is not None:
                if cv_mse < best_cv_mse:
                    best_cv_mse = cv_mse
                    best_predict = predict
                if mse < best_train_mse:
                    best_train_mse = mse
                    best_predict_train = predict
                    
        except Exception:
            continue
    
    # If no model worked, fallback to degree-4 polynomial
    if best_predict is None:
        Xf = _poly_features(X_train, 4)
        w, *_ = np.linalg.lstsq(Xf, y_train, rcond=None)
        best_predict = lambda X, w=w: _poly_features(X, 4) @ w
    
    return best_predict
# EVOLVE-BLOCK-END
