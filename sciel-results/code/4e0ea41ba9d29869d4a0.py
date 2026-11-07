"""Improved symbolic regression using Gaussian Process with Matérn kernel."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import warnings
from scipy.optimize import minimize
from scipy.linalg import cho_factor, cho_solve

_START_TIME = None
TIME_LIMIT = 300.0
SAFETY_MARGIN = 10.0


def _time_left():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
        return TIME_LIMIT - SAFETY_MARGIN
    return TIME_LIMIT - SAFETY_MARGIN - (time.time() - _START_TIME)


def _matern_kernel(X1, X2, l, nu=2.5):
    """
    Matérn kernel function.
    X1, X2: arrays of shape (n1, d), (n2, d)
    l: length scale
    nu: smoothness parameter
    """
    # Compute pairwise distances
    # Using broadcasting for efficiency
    # X1 shape (n1, 1, d), X2 shape (1, n2, d)
    diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]
    # Squared distances
    sq_dist = np.sum(diff ** 2, axis=-1)
    
    # Handle zero distances to avoid 0^nu issues if nu < 1, though here nu=2.5
    # The Matérn kernel for nu=2.5 is:
    # K(r) = 1 + sqrt(3)*r/l + (3*r^2)/(2*l^2) * exp(-sqrt(3)*r/l)
    # Wait, standard Matern formula:
    # K(r) = (2^(1-nu) / Gamma(nu)) * (sqrt(2*nu)/l * r)^nu * K_nu(sqrt(2*nu)/l * r)
    # For nu=2.5, this simplifies to:
    # K(r) = (1 + sqrt(3)*r/l + (3*r^2)/(2*l^2)) * exp(-sqrt(3)*r/l)
    
    r = np.sqrt(sq_dist)
    sqrt3_r_l = np.sqrt(3.0) * r / l
    # Avoid log(0) or issues if l is very small
    with np.errstate(divide='ignore', invalid='ignore'):
        term = (1.0 + sqrt3_r_l + 0.5 * sqrt3_r_l**2) * np.exp(-sqrt3_r_l)
    
    # If l is extremely small, off-diagonal terms go to 0, diagonal is 1.
    # If l is large, it's roughly 1.
    # We add small jitter later.
    return term


def _fit_gp(X_train, y_train, max_time=10.0):
    """
    Fit a Gaussian Process regression model.
    Returns a predict function.
    """
    n = X_train.shape[0]
    d = X_train.shape[1]
    
    # Normalize inputs to [0, 1] or centered to help optimization
    # Since we have bounds in the task description, we can use them, 
    # but we only have X_train. Let's use min/max of X_train for scaling.
    X_min = X_train.min(axis=0)
    X_max = X_train.max(axis=0)
    X_range = X_max - X_min
    # Avoid division by zero
    X_range[X_range == 0] = 1.0
    
    def scale(X):
        return (X - X_min) / X_range
    
    X_scaled = scale(X_train)
    
    # Hyperparameters: length_scale, sigma_f (signal std), sigma_n (noise std)
    # Log-transform for positivity
    def neg_log_marginal_likelihood(params):
        log_l, log_sigma_f, log_sigma_n = params
        l = np.exp(log_l)
        sigma_f = np.exp(log_sigma_f)
        sigma_n = np.exp(log_sigma_n)
        
        K = sigma_f**2 * _matern_kernel(X_scaled, X_scaled, l)
        K[np.diag_indices(n)] += sigma_n**2 + 1e-8 # Jitter
        
        try:
            L, info = cho_factor(K, lower=True, check_finite=False)
            if info > 0:
                return 1e10 # Infeasible
            alpha, _ = cho_solve((L, info), y_train, check_finite=False)
            # NLL = 0.5 * y^T K^-1 y + 0.5 * log det K + 0.5 * n * log(2*pi)
            log_det_K = 2.0 * np.sum(np.log(np.diag(L)))
            nll = 0.5 * np.dot(y_train, alpha) + 0.5 * log_det_K + 0.5 * n * np.log(2 * np.pi)
            return nll
        except Exception:
            return 1e10

    # Initial guesses
    # l ~ 0.1, sigma_f ~ std(y), sigma_n ~ 0.01
    init_log_l = np.log(0.5)
    init_log_sigma_f = np.log(max(np.std(y_train), 1e-3))
    init_log_sigma_n = np.log(1e-2)
    
    x0 = np.array([init_log_l, init_log_sigma_f, init_log_sigma_n])
    
    # Optimization
    best_x = x0
    best_nll = neg_log_marginal_likelihood(x0)
    
    # Use L-BFGS-B
    # We don't have a strict time limit per optimization step, but we can check overall time
    # Since we have 300s total and 4 equations, we have ~75s per equation.
    # GP optimization is fast for n=200.
    
    res = minimize(neg_log_marginal_likelihood, x0, method='L-BFGS-B', 
                   options={'maxiter': 100, 'maxfun': 2000})
    
    if res.fun < best_nll:
        best_x = res.x
        best_nll = res.fun
        
    log_l, log_sigma_f, log_sigma_n = best_x
    l = np.exp(log_l)
    sigma_f = np.exp(log_sigma_f)
    sigma_n = np.exp(log_sigma_n)
    
    K = sigma_f**2 * _matern_kernel(X_scaled, X_scaled, l)
    K[np.diag_indices(n)] += sigma_n**2 + 1e-8
    
    try:
        L, info = cho_factor(K, lower=True, check_finite=False)
        if info > 0:
            raise ValueError("Cholesky failed")
        alpha, _ = cho_solve((L, info), y_train, check_finite=False)
    except Exception:
        # Fallback to ridge if GP fails
        return _fit_ridge(X_train, y_train)

    def predict(X):
        X_scaled_test = scale(X)
        K_star = sigma_f**2 * _matern_kernel(X_scaled_test, X_scaled, l)
        mean = K_star @ alpha
        return mean

    return predict


def _fit_ridge(X_train, y_train):
    """Fallback: Ridge regression with polynomial features."""
    d = X_train.shape[1]
    def features(X):
        cols = [np.ones(len(X))]
        for j in range(d):
            x = X[:, j]
            for k in range(1, 5): # Degree 4
                cols.append(x ** k)
        return np.column_stack(cols)
    
    X_f = features(X_train)
    # Add regularization
    alpha = 1e-2
    A = X_f.T @ X_f + alpha * np.eye(X_f.shape[1])
    b = X_f.T @ y_train
    w = np.linalg.solve(A, b)
    
    def predict(X):
        return features(X) @ w
        
    return predict


def fit(X_train, y_train):
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
        
    remaining = _time_left()
    
    if remaining < 5.0:
        # Not enough time for GP, just use ridge
        return _fit_ridge(X_train, y_train)
        
    try:
        # GP is the main strategy
        predict_func = _fit_gp(X_train, y_train, max_time=remaining)
        
        # Sanity check: ensure predictions are finite
        test_pred = predict_func(X_train)
        if not np.all(np.isfinite(test_pred)):
            return _fit_ridge(X_train, y_train)
            
        return predict_func
    except Exception:
        # If GP fails for any reason, fallback
        return _fit_ridge(X_train, y_train)

# EVOLVE-BLOCK-END
