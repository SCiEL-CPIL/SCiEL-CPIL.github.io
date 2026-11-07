"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')


def _matern52_kernel(X1, X2, length_scale, sigma_f):
    """Compute Matern-5/2 kernel between two sets of points."""
    X1 = np.asarray(X1, dtype=np.float64)
    X2 = np.asarray(X2, dtype=np.float64)
    if X1.ndim == 1:
        X1 = X1.reshape(-1, 1)
    if X2.ndim == 1:
        X2 = X2.reshape(-1, 1)
    
    d2 = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * X1 @ X2.T
    d2 = np.maximum(d2, 0)
    d = np.sqrt(d2)
    
    sqrt3_d = np.sqrt(3) * d / length_scale
    K = sigma_f**2 * (1 + sqrt3_d + sqrt3_d**2 / 3) * np.exp(-sqrt3_d)
    
    if X1.shape[0] == X2.shape[0] and X1.shape == X2.shape:
        np.fill_diagonal(K, sigma_f**2)
    
    return K


def _neg_log_likelihood(params, X, y, sigma_n):
    """Negative log-likelihood for Gaussian Process."""
    length_scale = np.exp(params[0])
    sigma_f = np.exp(params[1])
    
    K = _matern52_kernel(X, X, length_scale, sigma_f)
    K[np.diag_indices_from(K)] += sigma_n**2
    
    try:
        L = np.linalg.cholesky(K + 1e-10 * np.eye(K.shape[0]))
    except np.linalg.LinAlgError:
        return 1e6
    
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
    nll = 0.5 * y @ alpha + np.sum(np.log(np.diag(L))) + 0.5 * len(y) * np.log(2 * np.pi)
    return nll


def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 280
    
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    
    n_samples, n_features = X_train.shape
    
    X_mean = np.mean(X_train, axis=0)
    X_std = np.std(X_train, axis=0)
    X_std[X_std < 1e-10] = 1.0
    X_train_std = (X_train - X_mean) / X_std
    
    y_mean = np.mean(y_train)
    y_var = np.var(y_train)
    sigma_n = max(0.01 * np.sqrt(y_var), 1e-3)
    
    init_params = [np.log(1.0), np.log(max(np.std(y_train), 0.1))]
    
    best_params = init_params
    best_nll = _neg_log_likelihood(init_params, X_train_std, y_train, sigma_n)
    
    start_points = [
        [np.log(0.5), np.log(max(np.std(y_train), 0.1))],
        [np.log(1.0), np.log(max(np.std(y_train), 0.1))],
        [np.log(2.0), np.log(max(np.std(y_train), 0.1))],
        [np.log(0.5), np.log(max(np.std(y_train) * 0.5, 0.1))],
        [np.log(2.0), np.log(max(np.std(y_train) * 2.0, 0.1))],
    ]
    
    for sp in start_points:
        if time.time() > deadline:
            break
        try:
            res = minimize(
                _neg_log_likelihood,
                sp,
                args=(X_train_std, y_train, sigma_n),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-6}
            )
            if res.fun < best_nll:
                best_nll = res.fun
                best_params = res.x
        except Exception:
            pass
    
    length_scale = np.exp(best_params[0])
    sigma_f = np.exp(best_params[1])
    
    K_train = _matern52_kernel(X_train_std, X_train_std, length_scale, sigma_f)
    K_train[np.diag_indices_from(K_train)] += sigma_n**2
    
    try:
        L = np.linalg.cholesky(K_train + 1e-10 * np.eye(K_train.shape[0]))
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
    except np.linalg.LinAlgError:
        def _poly_features(X):
            cols = [np.ones(len(X))] + [X[:, j] for j in range(X.shape[1])] + \
                   [X[:, i] * X[:, j] for i in range(X.shape[1]) for j in range(i, X.shape[1])]
            return np.column_stack(cols)
        w, *_ = np.linalg.lstsq(_poly_features(X_train), y_train, rcond=None)
        return lambda X: _poly_features(np.asarray(X, dtype=np.float64)) @ w
    
    # Capture variables for closure
    _X_mean = X_mean
    _X_std = X_std
    _X_train_std = X_train_std
    _length_scale = length_scale
    _sigma_f = sigma_f
    _alpha = alpha
    
    def predict(X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        X_std = (X - _X_mean) / _X_std
        
        K_star = _matern52_kernel(X_std, _X_train_std, _length_scale, _sigma_f)
        pred = K_star @ _alpha
        return pred
    
    return predict
# EVOLVE-BLOCK-END
