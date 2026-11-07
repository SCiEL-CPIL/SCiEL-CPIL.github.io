import numpy as np
import time
from scipy.interpolate import UnivariateSpline

_START_TIME = None
TIME_LIMIT = 330.0
SAFETY_MARGIN = 25.0

def _time_left():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
        return TIME_LIMIT - SAFETY_MARGIN
    return TIME_LIMIT - SAFETY_MARGIN - (time.time() - _START_TIME)

def _chebyshev_terms(x, max_deg):
    """Return list of Chebyshev polynomials T_0..T_max_deg evaluated at x."""
    n = x.shape[0]
    t0 = np.ones(n)
    t1 = x.copy()
    out = [t0, t1]
    for k in range(2, max_deg + 1):
        tk = 2.0 * x * t1 - t0
        out.append(tk)
        t0, t1 = t1, tk
    return out

def _build_rich_basis(x, max_cheb, n_fourier):
    """Build a rich basis: Chebyshev + Fourier terms for 1-D input."""
    n = x.shape[0]
    cols = []
    # Chebyshev polynomials (well-conditioned for polynomial approximation)
    for t in _chebyshev_terms(x, max_cheb):
        cols.append(t)
    # Fourier terms for periodic/oscillatory components
    for k in range(1, n_fourier + 1):
        cols.append(np.sin(k * x))
        cols.append(np.cos(k * x))
    return np.column_stack(cols)

def _loocv_ridge(B, y, lam):
    """Efficient LOOCV MSE for ridge regression using hat-matrix formula."""
    n, p = B.shape
    A = B.T @ B + lam * np.eye(p)
    try:
        A_inv = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        return np.inf
    B_Ainv = B @ A_inv
    h = np.einsum('ij,ij->i', B, B_Ainv)
    yhat = B @ (A_inv @ (B.T @ y))
    denom = 1.0 - h
    denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
    resid = y - yhat
    return np.mean((resid / denom) ** 2)

def _loocv_kernel(K, y, lam):
    """Efficient LOOCV MSE for kernel ridge regression."""
    n = K.shape[0]
    A = K + lam * np.eye(n)
    try:
        A_inv = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        return np.inf
    K_Ainv = K @ A_inv
    h = np.einsum('ij,ij->i', K, K_Ainv)
    yhat = K @ (A_inv @ y)
    denom = 1.0 - h
    denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
    resid = y - yhat
    return np.mean((resid / denom) ** 2)

def _rbf_kernel(X1, X2, sigma):
    """RBF kernel for 1-D data (X1, X2 have shape (n, 1))."""
    d2 = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * X1 @ X2.T
    d2 = np.maximum(d2, 0)
    return np.exp(-d2 / (2 * sigma**2))

def fit(X_train, y_train):
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
        
    X = X_train.astype(np.float64)
    y = y_train.astype(np.float64)
    n, d = X.shape
    x = X[:, 0].copy()
    
    # Standardize X
    mu = np.mean(x)
    sigma = np.std(x)
    if sigma == 0:
        sigma = 1.0
    x_scaled = (x - mu) / sigma
    
    lo, hi = float(np.min(x_scaled)), float(np.max(x_scaled))
    span = max(hi - lo, 1e-6)
    
    best_cv_error = np.inf
    best_predict_fn = None
    
    # --- Model 1: Rich basis (Chebyshev + Fourier) with ridge ---
    # Try different basis sizes
    basis_configs = [
        (10, 5), (12, 6), (15, 8), (18, 10), (20, 12), (25, 15)
    ]
    lam_grid = np.logspace(-8, 6, 20)
    
    for max_cheb, n_fourier in basis_configs:
        if _time_left() < 10.0:
            break
        B = _build_rich_basis(x_scaled, max_cheb, n_fourier)
        p = B.shape[1]
        for lam in lam_grid:
            if _time_left() < 5.0:
                break
            cv = _loocv_ridge(B, y, lam)
            if cv < best_cv_error:
                best_cv_error = cv
                # Refit on full data
                A = B.T @ B + lam * np.eye(p)
                try:
                    w = np.linalg.solve(A, B.T @ y)
                except np.linalg.LinAlgError:
                    w = np.linalg.lstsq(B, y, rcond=None)[0]
                w = np.where(np.isfinite(w), w, 0.0)
                
                def predict_rich(X, B_fn=_build_rich_basis, max_c=max_cheb, n_f=n_fourier, w=w, mu=mu, sigma=sigma):
                    X = np.asarray(X, dtype=np.float64)
                    if X.ndim == 1:
                        X = X.reshape(1, -1)
                    xv = (X[:, 0] - mu) / sigma
                    Bx = B_fn(xv, max_c, n_f)
                    out = Bx @ w
                    return np.where(np.isfinite(out), out, 0.0)
                
                best_predict_fn = predict_rich
    
    # --- Model 2: RBF kernel ridge ---
    sigma_candidates = [span * s for s in [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]]
    lam_grid_k = np.logspace(-6, 4, 15)
    
    for sigma in sigma_candidates:
        if _time_left() < 10.0:
            break
        X1 = x_scaled[:, None]
        K = _rbf_kernel(X1, X1, sigma)
        for lam in lam_grid_k:
            if _time_left() < 5.0:
                break
            cv = _loocv_kernel(K, y, lam)
            if cv < best_cv_error:
                best_cv_error = cv
                A = K + lam * np.eye(n)
                try:
                    alpha = np.linalg.solve(A, y)
                except np.linalg.LinAlgError:
                    alpha = np.linalg.lstsq(K, y, rcond=None)[0]
                alpha = np.where(np.isfinite(alpha), alpha, 0.0)
                
                K_train = K.copy()
                alpha_store = alpha.copy()
                
                def predict_rbf(X, K_tr=K_train, alpha=alpha_store, sig=sigma, mu=mu, sigma=sigma, kern=_rbf_kernel):
                    X = np.asarray(X, dtype=np.float64)
                    if X.ndim == 1:
                        X = X.reshape(1, -1)
                    xv = (X[:, 0] - mu) / sigma
                    K_val = kern(xv[:, None], K_tr[:, :0].shape[0] and np.ones((K_tr.shape[0], 1)), sig)
                    # Recompute properly
                    xv_col = xv[:, None]
                    # We need the training points for kernel evaluation
                    # Store them
                    return K_val @ alpha
                # Fix: store training x
                x_train_scaled_store = x_scaled.copy()
                
                def predict_rbf(X, x_tr=x_train_scaled_store, alpha=alpha_store, sig=sigma, mu=mu, sigma=sigma, kern=_rbf_kernel):
                    X = np.asarray(X, dtype=np.float64)
                    if X.ndim == 1:
                        X = X.reshape(1, -1)
                    xv = (X[:, 0] - mu) / sigma
                    K_val = kern(xv[:, None], x_tr[:, None], sig)
                    out = K_val @ alpha
                    return np.where(np.isfinite(out), out, 0.0)
                
                best_predict_fn = predict_rbf
    
    # --- Model 3: Spline ---
    s_candidates = np.logspace(-5, 5, 25)
    for s_val in s_candidates:
        if _time_left() < 5.0:
            break
        sort_idx = np.argsort(x_scaled)
        x_sorted = x_scaled[sort_idx]
        y_sorted = y[sort_idx]
        try:
            sp = UnivariateSpline(x_sorted, y_sorted, s=s_val, k=3)
            # Simple CV: hold out 20%
            n_val = max(20, n // 5)
            val_idx = np.random.choice(n, n_val, replace=False)
            train_idx = np.setdiff1d(np.arange(n), val_idx)
            sp_cv = UnivariateSpline(x_scaled[train_idx], y[train_idx], s=s_val, k=3)
            pred_val = sp_cv(x_scaled[val_idx])
            cv = np.mean((pred_val - y[val_idx])**2)
            if cv < best_cv_error:
                best_cv_error = cv
                sp_final = UnivariateSpline(x_sorted, y_sorted, s=s_val, k=3)
                
                def predict_spline(X, sp=sp_final, mu=mu, sigma=sigma):
                    X = np.asarray(X, dtype=np.float64)
                    if X.ndim == 1:
                        X = X.reshape(1, -1)
                    xv = (X[:, 0] - mu) / sigma
                    out = sp(xv)
                    return np.where(np.isfinite(out), out, 0.0)
                
                best_predict_fn = predict_spline
        except Exception:
            continue
    
    # --- Fallback ---
    if best_predict_fn is None:
        # Simple linear fit as ultimate fallback
        A = np.column_stack([np.ones(n), x_scaled])
        try:
            w = np.linalg.lstsq(A, y, rcond=None)[0]
        except:
            w = np.array([0.0, 0.0])
        
        def predict_linear(X, w=w, mu=mu, sigma=sigma):
            X = np.asarray(X, dtype=np.float64)
            if X.ndim == 1:
                X = X.reshape(1, -1)
            xv = (X[:, 0] - mu) / sigma
            return w[0] + w[1] * xv
    
    else:
        predict_linear = best_predict_fn
    
    # Final safety wrapper
    def predict(X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        try:
            out = predict_linear(X)
        except Exception:
            out = np.full(X.shape[0], np.mean(y))
        out = np.asarray(out, dtype=np.float64)
        out = np.where(np.isfinite(out), out, np.mean(y))
        return out
    
    return predict
