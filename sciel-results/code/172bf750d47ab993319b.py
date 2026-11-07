"""
Strategy: 1D kernel regression + ensemble of basis functions with careful CV.
Since all equations are 1D, we exploit this heavily. We use:
1. Gaussian kernel regression (RBF) with multiple bandwidths
2. Polynomial regression with degree selection
3. Fourier regression for periodic functions
4. Ensemble blending of top candidates
5. Time guard with safety margin
"""
import numpy as np
import time

def _rbf_kernel(X1, X2, bandwidth):
    """Compute RBF kernel matrix."""
    # X1: (n,1), X2: (m,1)
    d2 = (X1 - X2)**2
    return np.exp(-d2 / (2 * bandwidth**2))

def _fit_rbf(X, y, bandwidth, reg=1e-6):
    """Fit RBF kernel regression. Returns (centers, weights, bandwidth)."""
    n = len(X)
    K = _rbf_kernel(X, X, bandwidth)
    K[~np.isfinite(K)] = 0
    # Regularize
    K_reg = K + reg * np.eye(n)
    try:
        w = np.linalg.solve(K_reg, y)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(K_reg, y, rcond=None)[0]
    return X, w, bandwidth

def _predict_rbf(X_new, centers, weights, bandwidth):
    """Predict with RBF kernel regression."""
    K = _rbf_kernel(X_new, centers, bandwidth)
    K[~np.isfinite(K)] = 0
    return K @ weights

def _fit_poly(X, y, degree, reg=1e-6):
    """Fit polynomial regression. Returns (degree, coefficients)."""
    n = len(X)
    # Build Vandermonde-like matrix
    V = np.vander(X, N=degree+1, increasing=True)
    V[~np.isfinite(V)] = 0
    A = V.T @ V + reg * np.eye(degree+1)
    b = V.T @ y
    try:
        w = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(V, y, rcond=None)[0]
    return degree, w

def _predict_poly(X_new, degree, coeffs):
    """Predict with polynomial."""
    V = np.vander(X_new, N=degree+1, increasing=True)
    V[~np.isfinite(V)] = 0
    return V @ coeffs

def _fit_fourier(X, y, n_freqs, reg=1e-6):
    """Fit Fourier regression. Returns (n_freqs, coefficients)."""
    n = len(X)
    cols = [np.ones(n)]
    for k in range(1, n_freqs + 1):
        cols.append(np.sin(k * X))
        cols.append(np.cos(k * X))
    F = np.column_stack(cols)
    F[~np.isfinite(F)] = 0
    p = F.shape[1]
    A = F.T @ F + reg * np.eye(p)
    b = F.T @ y
    try:
        w = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(F, y, rcond=None)[0]
    return n_freqs, w

def _predict_fourier(X_new, n_freqs, coeffs):
    """Predict with Fourier."""
    n = len(X_new)
    cols = [np.ones(n)]
    for k in range(1, n_freqs + 1):
        cols.append(np.sin(k * X_new))
        cols.append(np.cos(k * X_new))
    F = np.column_stack(cols)
    F[~np.isfinite(F)] = 0
    return F @ coeffs

def _cross_validate_predictor(make_fit, make_predict, X, y, n_folds=5, **params):
    """K-fold CV for a fit/predict pair."""
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
        
        X_tr = X[train_idx]
        y_tr = y[train_idx]
        X_val = X[val_idx]
        y_val = y[val_idx]
        
        model = make_fit(X_tr, y_tr, **params)
        y_pred = make_predict(X_val, model, **params)
        y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=1e6, neginf=-1e6)
        mse = np.mean((y_pred - y_val)**2)
        mse_sum += mse
    
    return mse_sum / n_folds

def fit(X_train, y_train):
    start_time = time.time()
    deadline = 300.0  # seconds, with safety margin
    
    X = X_train.reshape(-1, 1).astype(np.float64)
    y = y_train.reshape(-1).astype(np.float64)
    n = len(X)
    
    # Sort by x for potential interpolation
    sort_idx = np.argsort(X[:, 0])
    X_sorted = X[sort_idx]
    y_sorted = y[sort_idx]
    
    x_min = np.min(X[:, 0])
    x_max = np.max(X[:, 0])
    x_range = x_max - x_min
    if x_range < 1e-10:
        x_range = 1.0
    
    candidates = []  # list of (name, cv_mse, model, predict_func)
    
    # 1. RBF with various bandwidths
    bandwidths = [x_range * f for f in [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]]
    for bw in bandwidths:
        if time.time() > deadline - 30:
            break
        try:
            cv_mse = _cross_validate_predictor(
                lambda X_, y_, **kw: _fit_rbf(X_, y_, bw),
                lambda X_, model, **kw: _predict_rbf(X_, model[0], model[1], model[2]),
                X, y, n_folds=5
            )
            model = _fit_rbf(X, y, bw)
            candidates.append(('rbf_bw%.4f' % bw, cv_mse, model, 
                              lambda X_new, m=model: _predict_rbf(X_new, m[0], m[1], m[2])))
        except Exception:
            continue
    
    # 2. Polynomial with various degrees
    for deg in [2, 3, 4, 5, 6, 7, 8]:
        if time.time() > deadline - 30:
            break
        try:
            cv_mse = _cross_validate_predictor(
                lambda X_, y_, **kw: _fit_poly(X_, y_, deg),
                lambda X_, model, **kw: _predict_poly(X_, model[0], model[1]),
                X, y, n_folds=5
            )
            model = _fit_poly(X, y, deg)
            candidates.append(('poly_deg%d' % deg, cv_mse, model,
                              lambda X_new, m=model: _predict_poly(X_new, m[0], m[1])))
        except Exception:
            continue
    
    # 3. Fourier with various frequencies
    for nf in [2, 3, 4, 5, 6, 8, 10]:
        if time.time() > deadline - 30:
            break
        try:
            cv_mse = _cross_validate_predictor(
                lambda X_, y_, **kw: _fit_fourier(X_, y_, nf),
                lambda X_, model, **kw: _predict_fourier(X_, model[0], model[1]),
                X, y, n_folds=5
            )
            model = _fit_fourier(X, y, nf)
            candidates.append(('fourier_nf%d' % nf, cv_mse, model,
                              lambda X_new, m=model: _predict_fourier(X_new, m[0], m[1])))
        except Exception:
            continue
    
    if not candidates:
        # Fallback: simple linear
        def predict(X_new):
            X_new = X_new.reshape(-1, 1)
            slope = np.mean(X[:, 0] * y) / (np.mean(X[:, 0]**2) + 1e-10)
            intercept = np.mean(y) - slope * np.mean(X[:, 0])
            return slope * X_new[:, 0] + intercept
        return predict
    
    # Sort by CV MSE
    candidates.sort(key=lambda c: c[1])
    
    # Ensemble: blend top K candidates weighted by inverse CV MSE
    top_k = min(5, len(candidates))
    top_candidates = candidates[:top_k]
    
    # Inverse MSE weights (with floor to avoid division by zero)
    mses = np.array([c[1] for c in top_candidates])
    mses = np.maximum(mses, 1e-12)
    weights = 1.0 / mses
    weights = weights / np.sum(weights)
    
    def predict(X_new):
        X_new = X_new.reshape(-1, 1).astype(np.float64)
        pred_sum = np.zeros(X_new.shape[0])
        for i, (name, cv_mse, model, pred_func) in enumerate(top_candidates):
            try:
                p = pred_func(X_new)
                p = np.nan_to_num(p, nan=0.0, posinf=1e6, neginf=-1e6)
                pred_sum += weights[i] * p
            except Exception:
                continue
        return pred_sum
    
    return predict
