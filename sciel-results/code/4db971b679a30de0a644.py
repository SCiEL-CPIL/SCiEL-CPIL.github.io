"""Symbolic regression over 10 hidden equations: Multi-stage oracle with Poly-Kernel, GP, Rational."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.linalg import solve

_START = time.time()
_DEADLINE = 295.0

def _remaining():
    return _DEADLINE - (time.time() - _START)

def _poly_kernel(X1, X2, degree):
    """Polynomial kernel: (X1·X2 + 1)^degree"""
    return (X1 @ X2.T + 1.0) ** degree

def _matern52_kernel(X1, X2, length_scale, variance):
    """Matérn 5/2 kernel."""
    d1 = np.sqrt(np.sum(X1 ** 2, axis=1))
    d2 = np.sqrt(np.sum(X2 ** 2, axis=1))
    dist = np.sqrt(np.maximum(d1[:, None] ** 2 + d2[None, :] ** 2 - 2 * X1 @ X2.T, 0.0))
    r = dist / length_scale
    return variance * (1.0 + np.sqrt(5) * r + 5.0 / 3.0 * r ** 2) * np.exp(-np.sqrt(5) * r)

def _fit_poly_kernel(X, y, degree, alphas):
    """Fit polynomial kernel ridge regression. Returns (predict_fn, train_mse)."""
    n = X.shape[0]
    K = _poly_kernel(X, X, degree)
    best_pred = None
    best_mse = float('inf')
    for alpha in alphas:
        try:
            K_reg = K + alpha * np.eye(n)
            w = solve(K_reg, y)
            pred = K @ w
            if not np.all(np.isfinite(pred)):
                continue
            mse = np.mean((pred - y) ** 2)
            if mse < best_mse:
                best_mse = mse
                best_pred = w.copy()
        except Exception:
            continue
    if best_pred is None:
        return None, float('inf')
    K_train = K
    w_final = best_pred
    def predict(X_new):
        X_new = np.asarray(X_new, dtype=float)
        if X_new.ndim == 1:
            X_new = X_new.reshape(-1, 1)
        K_new = _poly_kernel(X_new, X, degree)
        return K_new @ w_final
    return predict, best_mse

def _fit_gp(X, y, length_scales, variances, noise_levels):
    """Fit Gaussian Process with Matérn 5/2 kernel. Returns (predict_fn, train_mse)."""
    n = X.shape[0]
    best_pred = None
    best_mse = float('inf')
    for ls in length_scales:
        for var in variances:
            for noise in noise_levels:
                try:
                    K = _matern52_kernel(X, X, ls, var) + noise * np.eye(n)
                    L = np.linalg.cholesky(K + 1e-10 * np.eye(n))
                    alpha_vec = np.linalg.solve(L.T, np.linalg.solve(L, y))
                    pred = K @ alpha_vec
                    if not np.all(np.isfinite(pred)):
                        continue
                    mse = np.mean((pred - y) ** 2)
                    if mse < best_mse:
                        best_mse = mse
                        # Store for prediction
                        K_train_stored = K.copy()
                        alpha_stored = alpha_vec.copy()
                except Exception:
                    continue
    if best_pred is None and 'K_train_stored' not in dir():
        # Try simpler fallback
        K = _matern52_kernel(X, X, 1.0, 1.0) + 0.01 * np.eye(n)
        try:
            w = solve(K, y)
            pred = K @ w
            mse = np.mean((pred - y) ** 2)
            if np.all(np.isfinite(pred)):
                K_train_stored = K
                alpha_stored = w
                best_mse = mse
            else:
                return None, float('inf')
        except Exception:
            return None, float('inf')
    K_train_stored = K_train_stored
    alpha_stored = alpha_stored
    def predict(X_new):
        X_new = np.asarray(X_new, dtype=float)
        if X_new.ndim == 1:
            X_new = X_new.reshape(-1, 1)
        K_new = _matern52_kernel(X_new, X, 1.0, 1.0)
        return K_new @ alpha_stored
    return predict, best_mse

def _fit_rational(X, y, n_terms):
    """Fit rational function: N(X)/D(X) where N, D are linear in features.
    Uses iterative nonlinear least squares."""
    n, d = X.shape
    # Features: [1, x1, ..., xd]
    F = np.column_stack([np.ones(n), X])
    p = F.shape[1]
    
    # Initialize D = 1 (so rational = linear)
    D = np.ones(n)
    best_mse = float('inf')
    best_num = None
    best_den = None
    
    for iteration in range(20):
        try:
            # Solve for numerator given denominator
            Fw = F * D[:, None]
            num_coeffs, *_ = np.linalg.lstsq(Fw, y, rcond=None)
            N = F @ num_coeffs
            
            # Solve for denominator given numerator
            # Minimize ||N - y*D||^2 => D minimizes ||F^T(N-yF)|| ... 
            # Actually: minimize sum (N_i - y_i * D_i)^2
            # d/dD: -2*F^T*(N - y*D) = 0 => F^T F D = F^T N
            FF = F.T @ F
            FN = F.T @ N
            reg = 1e-6 * np.eye(p)
            den_coeffs = solve(FF + reg, FN)
            D_new = F @ den_coeffs
            D_new = np.where(np.abs(D_new) < 1e-8, np.sign(D_new) * 1e-8 + (D_new == 0) * 1e-8, D_new)
            
            N_new = F @ num_coeffs
            pred = N_new / D_new
            if not np.all(np.isfinite(pred)):
                break
            mse = np.mean((pred - y) ** 2)
            if mse < best_mse:
                best_mse = mse
                best_num = num_coeffs.copy()
                best_den = den_coeffs.copy()
            
            if np.max(np.abs(D_new - D)) < 1e-6:
                D = D_new
                break
            D = D_new
        except Exception:
            break
    
    if best_num is None:
        # Fallback to linear
        coeffs, *_ = np.linalg.lstsq(F, y, rcond=None)
        def predict(X_new):
            X_new = np.asarray(X_new, dtype=float)
            if X_new.ndim == 1:
                X_new = X_new.reshape(-1, 1)
            F_new = np.column_stack([np.ones(X_new.shape[0]), X_new])
            return F_new @ coeffs
        return predict, np.mean(((F @ coeffs) - y) ** 2)
    
    num_coeffs_final = best_num
    den_coeffs_final = best_den
    def predict(X_new):
        X_new = np.asarray(X_new, dtype=float)
        if X_new.ndim == 1:
            X_new = X_new.reshape(-1, 1)
        F_new = np.column_stack([np.ones(X_new.shape[0]), X_new])
        N = F_new @ num_coeffs_final
        D = F_new @ den_coeffs_final
        D = np.where(np.abs(D) < 1e-8, np.sign(D) * 1e-8 + (D == 0) * 1e-8, D)
        return N / D
    return predict, best_mse

def _loo_cv_mse(predict_fn, X, y):
    """Leave-one-out cross-validation MSE."""
    n = len(y)
    preds = np.zeros(n)
    for i in range(n):
        X_train_loo = np.delete(X, i, axis=0)
        y_train_loo = np.delete(y, i)
        try:
            # Refit on LOO data
            p_loo, _ = _fit_single_model(predict_fn.__name__ if hasattr(predict_fn, '__name__') else 'generic', X_train_loo, y_train_loo)
            if p_loo is not None:
                preds[i] = p_loo(X[i:i+1])[0]
            else:
                preds[i] = np.mean(y_train_loo)
        except Exception:
            preds[i] = np.mean(y)
    mse = np.mean((preds - y) ** 2)
    return mse

def _fit_single_model(model_type, X, y):
    """Dispatch to the appropriate model fitter."""
    if model_type == 'poly2':
        return _fit_poly_kernel(X, y, 2, [1e-6, 1e-4, 1e-3, 1e-2, 0.1, 1.0])
    elif model_type == 'poly3':
        return _fit_poly_kernel(X, y, 3, [1e-6, 1e-4, 1e-3, 1e-2, 0.1, 1.0])
    elif model_type == 'poly4':
        return _fit_poly_kernel(X, y, 4, [1e-6, 1e-4, 1e-3, 1e-2, 0.1, 1.0])
    elif model_type == 'gp':
        return _fit_gp(X, y, [0.5, 1.0, 2.0, 5.0], [0.1, 1.0, 10.0], [1e-4, 1e-3, 1e-2])
    elif model_type == 'rational':
        return _fit_rational(X, y, X.shape[1])
    else:
        return None, float('inf')

def fit(X_train, y_train):
    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    y = y.reshape(-1)
    n = X.shape[0]
    
    if _remaining() < 5.0:
        # Emergency fallback: simple linear
        F = np.column_stack([np.ones(n), X])
        w, *_ = np.linalg.lstsq(F, y, rcond=None)
        def predict(X):
            X = np.asarray(X, dtype=float)
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            F_new = np.column_stack([np.ones(X.shape[0]), X])
            return F_new @ w
        return predict
    
    # Fit diverse models
    models = {}
    
    # Polynomial kernel degree 2
    if _remaining() > 30:
        p, mse = _fit_poly_kernel(X, y, 2, [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0])
        if p is not None:
            models['poly2'] = (p, mse)
    
    # Polynomial kernel degree 3
    if _remaining() > 25:
        p, mse = _fit_poly_kernel(X, y, 3, [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0])
        if p is not None:
            models['poly3'] = (p, mse)
    
    # Polynomial kernel degree 4
    if _remaining() > 20:
        p, mse = _fit_poly_kernel(X, y, 4, [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0])
        if p is not None:
            models['poly4'] = (p, mse)
    
    # Gaussian Process
    if _remaining() > 15:
        p, mse = _fit_gp(X, y, [0.5, 1.0, 2.0, 5.0], [0.1, 1.0, 10.0], [1e-4, 1e-3, 1e-2])
        if p is not None:
            models['gp'] = (p, mse)
    
    # Rational function
    if _remaining() > 10:
        p, mse = _fit_rational(X, y, X.shape[1])
        if p is not None:
            models['rational'] = (p, mse)
    
    # Linear baseline
    F = np.column_stack([np.ones(n), X])
    w_lin, *_ = np.linalg.lstsq(F, y, rcond=None)
    def predict_lin(X_new):
        X_new = np.asarray(X_new, dtype=float)
        if X_new.ndim == 1:
            X_new = X_new.reshape(-1, 1)
        F_new = np.column_stack([np.ones(X_new.shape[0]), X_new])
        return F_new @ w_lin
    mse_lin = np.mean(((F @ w_lin) - y) ** 2)
    models['linear'] = (predict_lin, mse_lin)
    
    if not models:
        # Should not happen, but just in case
        return predict_lin
    
    # Select best model based on training MSE (proxy for generalization)
    # In a full implementation, we'd use LOO-CV, but that's expensive
    best_name = min(models.keys(), key=lambda k: models[k][1])
    best_predict = models[best_name][0]
    
    return best_predict
# EVOLVE-BLOCK-END
