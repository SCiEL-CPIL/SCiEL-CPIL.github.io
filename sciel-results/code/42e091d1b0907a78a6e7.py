"""Symbolic regression via adaptive feature selection and Gaussian Process hybrid."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import norm
import warnings

warnings.filterwarnings('ignore')

def _generate_feature_library(x):
    """
    Generate a diverse library of candidate features for 1D input x.
    Includes polynomials, trigonometric, exponential, and hybrid functions.
    """
    x = np.asarray(x, dtype=float)
    features = [np.ones_like(x)]
    names = ['1']
    
    # Polynomials up to degree 5
    for i in range(1, 6):
        f = x ** i
        features.append(f)
        names.append(f'x^{i}')
    
    # Trigonometric
    for i in range(1, 4):
        features.append(np.sin(i * x))
        names.append(f'sin({i}x)')
        features.append(np.cos(i * x))
        names.append(f'cos({i}x)')
    
    # Exponential and log-like (careful with domain)
    # exp(x) can blow up, so we clip or use scaled versions
    features.append(np.exp(0.5 * x))
    names.append('exp(0.5x)')
    features.append(np.exp(-0.5 * x))
    names.append('exp(-0.5x)')
    
    # Sqrt and reciprocal (safe versions)
    safe_x = np.where(np.abs(x) > 1e-8, x, 1e-8)
    features.append(np.sqrt(np.abs(x)))
    names.append('sqrt(|x|)')
    features.append(1.0 / safe_x)
    names.append('1/x')
    
    # Hyperbolic
    features.append(np.tanh(x))
    names.append('tanh(x)')
    features.append(np.sinh(x))
    names.append('sinh(x)')
    
    # Products of x with trig/exp
    features.append(x * np.sin(x))
    names.append('x*sin(x)')
    features.append(x * np.cos(x))
    names.append('x*cos(x)')
    features.append(x * np.exp(0.5 * x))
    names.append('x*exp(0.5x)')
    
    return np.column_stack(features)

def _greedy_forward_select(X_train, y_train, max_features=8, time_limit=10.0):
    """
    Greedy forward selection to pick the best subset of features.
    Returns the indices of selected features.
    """
    start_time = time.time()
    lib = _generate_feature_library(X_train)
    n_samples, n_features = lib.shape
    
    selected = []
    remaining = list(range(n_features))
    y_mean = np.mean(y_train)
    
    # Always include intercept
    selected.append(0)
    remaining.remove(0)
    
    while len(selected) < max_features and remaining:
        best_idx = -1
        best_mse = np.inf
        
        for idx in remaining:
            current = selected + [idx]
            X_sub = lib[:, current]
            try:
                w, *_ = np.linalg.lstsq(X_sub, y_train, rcond=None)
                pred = X_sub @ w
                mse = np.mean((pred - y_train) ** 2)
                if mse < best_mse:
                    best_mse = mse
                    best_idx = idx
            except:
                continue
        
        if best_idx == -1:
            break
            
        selected.append(best_idx)
        remaining.remove(best_idx)
        
        # Check time
        if time.time() - start_time > time_limit:
            break
            
    return selected

def _matern52_kernel(X1, X2, length_scale, signal_var):
    """Matérn 5/2 kernel for 1D inputs."""
    X1 = X1.reshape(-1, 1)
    X2 = X2.reshape(-1, 1)
    dist = np.sqrt(np.sum((X1 - X2.T) ** 2, axis=2))
    sqrt3_dist = np.sqrt(3.0) * dist / length_scale
    K = signal_var * (1.0 + sqrt3_dist) * np.exp(-sqrt3_dist)
    return K

def _log_likelihood(params, X, y):
    """Negative log marginal likelihood for GP."""
    length_scale, signal_var, noise_var = params
    length_scale = np.exp(length_scale)
    signal_var = np.exp(signal_var)
    noise_var = np.exp(noise_var)
    
    K = _matern52_kernel(X, X, length_scale, signal_var) + noise_var * np.eye(len(X))
    
    try:
        L = np.linalg.cholesky(K)
    except:
        return 1e10
    
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
    log_det = 2.0 * np.sum(np.log(np.diag(L)))
    nll = 0.5 * y.dot(alpha) + 0.5 * log_det + 0.5 * len(X) * np.log(2 * np.pi)
    return nll

def _fit_gp(X_train, y_train, time_limit=15.0):
    """Fit a GP with Matérn 5/2 kernel using grid search + local optimization."""
    start_time = time.time()
    
    # Normalize inputs and outputs for numerical stability
    x_mean = np.mean(X_train)
    x_std = np.std(X_train)
    if x_std < 1e-8:
        x_std = 1.0
    y_mean = np.mean(y_train)
    y_std = np.std(y_train)
    if y_std < 1e-8:
        y_std = 1.0
    
    X_norm = (X_train - x_mean) / x_std
    y_norm = (y_train - y_mean) / y_std
    
    # Grid search over log-space hyperparameters
    ls_candidates = np.logspace(-2, 2, 20)
    sv_candidates = np.logspace(-2, 2, 20)
    nv_candidates = np.logspace(-8, -2, 10)
    
    best_nll = np.inf
    best_params = None
    
    # Coarse grid search
    for ls in ls_candidates:
        for sv in sv_candidates:
            for nv in nv_candidates:
                nll = _log_likelihood([ls, sv, nv], X_norm, y_norm)
                if nll < best_nll:
                    best_nll = nll
                    best_params = [ls, sv, nv]
        
        if time.time() - start_time > time_limit * 0.5:
            break
    
    # Local optimization from best grid point
    if best_params is not None:
        try:
            res = minimize(_log_likelihood, best_params, args=(X_norm, y_norm), method='Nelder-Mead',
                           options={'maxiter': 100, 'xatol': 1e-4, 'fatol': 1e-4})
            if res.fun < best_nll:
                best_nll = res.fun
                best_params = res.x
        except:
            pass
    
    # Final GP fit with best hyperparameters
    length_scale = np.exp(best_params[0])
    signal_var = np.exp(best_params[1])
    noise_var = np.exp(best_params[2])
    
    K = _matern52_kernel(X_norm, X_norm, length_scale, signal_var) + noise_var * np.eye(len(X_norm))
    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_norm))
    
    # Define predict function
    def predict(X):
        X = np.asarray(X, dtype=float).reshape(-1)
        X_norm = (X - x_mean) / x_std
        K_star = _matern52_kernel(X_norm, X_norm, length_scale, signal_var)
        # Mean prediction
        mean = K_star @ alpha
        # Denormalize
        return mean * y_std + y_mean
    
    return predict, best_nll

def fit(X_train, y_train):
    """
    Hybrid regression: combines linear model on selected features with GP residual correction.
    """
    start_time = time.time()
    global_deadline = start_time + 280.0  # Safety margin before 300s limit
    
    X_train = np.asarray(X_train, dtype=float).reshape(-1)
    y_train = np.asarray(y_train, dtype=float).reshape(-1)
    
    # Step 1: Greedy forward selection for linear part
    selected_indices = _greedy_forward_select(X_train, y_train, max_features=8, time_limit=10.0)
    
    # Build linear model
    lib = _generate_feature_library(X_train)
    X_lin = lib[:, selected_indices]
    w, *_ = np.linalg.lstsq(X_lin, y_train, rcond=None)
    y_lin_pred = X_lin @ w
    residuals = y_train - y_lin_pred
    
    # Step 2: Fit GP on residuals if time allows
    gp_predict = None
    if time.time() < global_deadline - 20.0:
        try:
            gp_predict, gp_nll = _fit_gp(X_train, residuals, time_limit=15.0)
        except:
            gp_predict = None
    
    # Define combined predictor
    def predict(X):
        X = np.asarray(X, dtype=float).reshape(-1)
        lib_test = _generate_feature_library(X)
        y_lin = lib_test[:, selected_indices] @ w
        if gp_predict is not None:
            y_res = gp_predict(X)
        else:
            y_res = 0.0
        return y_lin + y_res
    
    return predict
# EVOLVE-BLOCK-END
