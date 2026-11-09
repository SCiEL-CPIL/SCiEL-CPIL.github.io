"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize_scalar
from scipy.spatial.distance import cdist

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 280  # Safety margin

    X = np.asarray(X_train, dtype=float).reshape(-1, 1)
    y = np.asarray(y_train, dtype=float).flatten()
    n = len(y)
    
    x_range = np.max(X) - np.min(X)
    if x_range <= 0:
        x_range = 1.0
    y_var = np.var(y)
    if y_var < 1e-10:
        y_var = 1.0
    y_mean = np.mean(y)
    
    # Normalize y for numerical stability in GP
    y_std = np.std(y)
    if y_std < 1e-10:
        y_std = 1.0
    y_norm = (y - y_mean) / y_std
    
    # Kernel: RBF (Gaussian) is often better for smooth 1D functions than Matern
    def rbf_kernel(X1, X2, length_scale):
        d = cdist(X1, X2, 'euclidean')
        return np.exp(-0.5 * (d / length_scale) ** 2)

    # We fix noise and variance, only optimize length scale
    noise = 1e-8 * y_std**2 # Very small noise for noiseless data
    variance = y_std**2
    
    # Precompute K for training set? No, length scale changes.
    
    def neg_log_marginal_likelihood(log_ls):
        if time.time() > deadline:
            return 1e10
        ls = np.exp(log_ls)
        if ls <= 0:
            return 1e10
        
        K = rbf_kernel(X, X, ls)
        K += noise * np.eye(n)
        
        try:
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_norm))
            log_det = 2 * np.sum(np.log(np.diag(L)))
            lml = -0.5 * y_norm.dot(alpha) - 0.5 * log_det - 0.5 * n * np.log(2 * np.pi)
            return -lml
        except np.linalg.LinAlgError:
            return 1e10

    # Coarse grid search for length scale
    # Range of length scales: from very small (interpolation) to very large (constant)
    ls_min = x_range * 0.01
    ls_max = x_range * 10.0
    
    log_ls_min = np.log(ls_min)
    log_ls_max = np.log(ls_max)
    
    # 20 points in log space
    grid = np.linspace(log_ls_min, log_ls_max, 20)
    best_nll = 1e10
    best_log_ls = grid[0]
    
    for log_ls in grid:
        nll = neg_log_marginal_likelihood(log_ls)
        if nll < best_nll:
            best_nll = nll
            best_log_ls = log_ls
            
    # Refine using minimize_scalar
    if time.time() < deadline - 10:
        try:
            res = minimize_scalar(neg_log_marginal_likelihood, bounds=(log_ls_min, log_ls_max), method='bounded', options={'xatol': 1e-6})
            if res.fun < best_nll:
                best_nll = res.fun
                best_log_ls = res.x
        except Exception:
            pass

    length_scale = np.exp(best_log_ls)
    
    # Final GP fit
    K_train = rbf_kernel(X, X, length_scale)
    K_train += noise * np.eye(n)
    
    try:
        L = np.linalg.cholesky(K_train)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_norm))
    except np.linalg.LinAlgError:
        # Fallback: PCHIP interpolation
        from scipy.interpolate import PchipInterpolator
        # Sort X for PCHIP
        sort_idx = np.argsort(X.flatten())
        X_sorted = X[sort_idx].flatten()
        y_sorted = y[sort_idx]
        pchip_interp = PchipInterpolator(X_sorted, y_sorted)
        def predict(X_pred):
            X_pred = np.asarray(X_pred, dtype=float).flatten()
            return pchip_interp(X_pred)
        return predict

    def predict(X_pred):
        X_pred = np.asarray(X_pred, dtype=float).reshape(-1, 1)
        K_star = rbf_kernel(X_pred, X, length_scale)
        mu_norm = K_star.dot(alpha)
        mu = mu_norm * y_std + y_mean
        mu = np.where(np.isfinite(mu), mu, 0.0)
        return mu

    return predict
# EVOLVE-BLOCK-END
