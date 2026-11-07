# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.linalg import cho_factor, cho_solve

_START_TIME = time.time()
_DEADLINE = _START_TIME + 320  # safety margin before 330s kill


def _time_left():
    return _DEADLINE - time.time()


def _matern52_kernel(X1, X2, length_scales, variance):
    """Matérn 5/2 kernel with ARD length scales.
    X1: (N1, d), X2: (N2, d), length_scales: (d,), variance: scalar
    """
    # Standardize by length scales
    X1_s = X1 / length_scales
    X2_s = X2 / length_scales
    # Euclidean distances
    D = cdist(X1_s, X2_s, metric='euclidean')
    r = D
    # Matérn 5/2: K = variance * (1 + sqrt(5)*r + 5/3*r^2) * exp(-sqrt(5)*r)
    sqrt5_r = np.sqrt(5.0) * r
    K = variance * (1.0 + sqrt5_r + (5.0 / 3.0) * r ** 2) * np.exp(-sqrt5_r)
    return K


def _log_marginal_likelihood(params, X, y, noise_var):
    """Negative log marginal likelihood for optimization.
    params: [log_length_scales (d), log_variance, log_noise_var]
    """
    d = X.shape[1]
    log_ls = params[:d]
    log_var = params[d]
    log_noise = params[d + 1]
    
    length_scales = np.exp(log_ls)
    variance = np.exp(log_var)
    noise_var = np.exp(log_noise)
    
    # Ensure numerical stability
    if length_scales.min() < 1e-6 or variance < 1e-6 or noise_var < 1e-6:
        return 1e10
    
    K = _matern52_kernel(X, X, length_scales, variance)
    K[np.diag_indices_from(K)] += noise_var
    
    # Compute log det and alpha via Cholesky
    try:
        L, lower = cho_factor(K, lower=True, check_finite=False)
        log_det = 2.0 * np.sum(np.log(np.diag(L)))
        alpha, _ = cho_solve((L, lower), y, check_finite=False)
    except np.linalg.LinAlgError:
        return 1e10
    
    # Negative log marginal likelihood
    nll = 0.5 * y.dot(alpha) + 0.5 * log_det + 0.5 * len(y) * np.log(2 * np.pi)
    return nll


def _fit_gp(X, y, time_budget=20.0):
    """Fit a Gaussian Process with Matérn 5/2 kernel and ARD.
    Returns (length_scales, variance, noise_var, L, alpha, y_mean)
    """
    start = time.time()
    N, d = X.shape
    
    # Standardize y for optimization stability
    y_mean = np.mean(y)
    y_std = np.std(y)
    if y_std < 1e-10:
        y_std = 1.0
    y_norm = (y - y_mean) / y_std
    
    # Initial hyperparameters
    # Length scales: start at 1.0 (in standardized space)
    log_ls_init = np.zeros(d)
    # Variance: start at 1.0
    log_var_init = 0.0
    # Noise: start small
    log_noise_init = np.log(1e-4)
    
    params_init = np.concatenate([log_ls_init, [log_var_init, log_noise_init]])
    
    bounds = [(-5.0, 5.0)] * d + [(-5.0, 5.0), (-10.0, 0.0)]
    
    # Optimize negative log marginal likelihood
    best_params = params_init
    best_nll = _log_marginal_likelihood(params_init, X, y_norm, 0.0)
    
    # Try multiple restarts if time permits
    n_restarts = 3
    rng = np.random.RandomState(42)
    
    for i in range(n_restarts):
        if time.time() - start > time_budget * 0.8:
            break
        
        if i == 0:
            params = params_init.copy()
        else:
            # Perturb initial params
            log_ls_perturb = rng.normal(0, 0.5, d)
            log_var_perturb = rng.normal(0, 0.5)
            log_noise_perturb = rng.normal(0, 1.0)
            params = np.concatenate([log_ls_perturb, [log_var_perturb, log_noise_perturb]])
            # Clip to bounds
            for j in range(d):
                params[j] = np.clip(params[j], bounds[j][0], bounds[j][1])
            params[d] = np.clip(params[d], bounds[d][0], bounds[d][1])
            params[d+1] = np.clip(params[d+1], bounds[d+1][0], bounds[d+1][1])
        
        res = minimize(
            _log_marginal_likelihood,
            params,
            args=(X, y_norm, 0.0),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6}
        )
        
        if res.fun < best_nll:
            best_nll = res.fun
            best_params = res.x
    
    # Extract final hyperparameters
    log_ls = best_params[:d]
    log_var = best_params[d]
    log_noise = best_params[d + 1]
    
    length_scales = np.exp(log_ls)
    variance = np.exp(log_var)
    noise_var = np.exp(log_noise)
    
    # Ensure minimum values
    length_scales = np.maximum(length_scales, 1e-4)
    variance = max(variance, 1e-4)
    noise_var = max(noise_var, 1e-8)
    
    # Compute Cholesky and alpha for prediction
    K = _matern52_kernel(X, X, length_scales, variance)
    K[np.diag_indices_from(K)] += noise_var
    
    try:
        L, lower = cho_factor(K, lower=True, check_finite=False)
        alpha, _ = cho_solve((L, lower), y_norm, check_finite=False)
    except np.linalg.LinAlgError:
        # Fallback: use identity-like kernel
        length_scales = np.ones(d)
        variance = 1.0
        noise_var = 1e-4
        K = _matern52_kernel(X, X, length_scales, variance)
        K[np.diag_indices_from(K)] += noise_var
        L, lower = cho_factor(K + 1e-8 * np.eye(N), lower=True, check_finite=False)
        alpha, _ = cho_solve((L, lower), y_norm, check_finite=False)
    
    return length_scales, variance, noise_var, L, lower, alpha, y_mean, y_std


def fit(X_train, y_train):
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    N, d = X_train.shape
    
    if N == 0:
        return lambda X: np.zeros(len(X))
    
    # Determine time budget
    time_budget = min(25.0, max(5.0, _time_left() / 10.0))
    
    # Fit GP
    length_scales, variance, noise_var, L, lower, alpha, y_mean, y_std = _fit_gp(
        X_train, y_train, time_budget=time_budget
    )
    
    # Capture for prediction
    _X_train = X_train
    _length_scales = length_scales
    _variance = variance
    _noise_var = noise_var
    _L = L
    _lower = lower
    _alpha = alpha
    _y_mean = y_mean
    _y_std = y_std
    
    def predict(X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Compute kernel between test and training points
        K_star = _matern52_kernel(X, _X_train, _length_scales, _variance)
        
        # Predict: y_pred = K_star @ alpha
        y_pred_norm = K_star.dot(_alpha)
        
        # Unstandardize
        y_pred = y_pred_norm * _y_std + _y_mean
        
        return y_pred
    
    return predict
# EVOLVE-BLOCK-END
