"""Symbolic regression with adaptive Gaussian-RBF basis and cross-validation model selection."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import warnings
from scipy.optimize import minimize

warnings.filterwarnings('ignore')


def _safe_predict(predict_fn, X):
    """Ensure predictions are finite."""
    X = np.asarray(X, dtype=float)
    try:
        pred = predict_fn(X)
    except Exception:
        return np.zeros(len(np.asarray(X).reshape(-1)))
    pred = np.asarray(pred, dtype=float)
    pred = np.where(np.isfinite(pred), pred, 0.0)
    return pred


def _gp_predict(X_pred, x_train, y_train, length_scale, sigma_f, alpha):
    """GP prediction using Cholesky decomposition."""
    x_train = np.asarray(x_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    X_pred = np.asarray(X_pred, dtype=float)
    
    n = len(x_train)
    n_pred = len(X_pred)
    
    # Kernel matrix K = sigma_f^2 * exp(-0.5 * (x_i - x_j)^2 / length_scale^2)
    diff = x_train[:, None] - x_train[None, :]
    K = sigma_f ** 2 * np.exp(-0.5 * diff ** 2 / (length_scale ** 2))
    K += alpha ** 2 * np.eye(n)
    
    # Cholesky decomposition
    try:
        L = np.linalg.cholesky(K)
    except np.linalg.LinAlgError:
        # Fallback: add jitter
        K += 1e-10 * np.eye(n)
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            # Final fallback: use pseudo-inverse
            alpha_inv = np.linalg.pinv(K)
            alpha_vec = alpha_inv @ y_train
            # k* = K(x_pred, x_train)
            diff_pred = X_pred[:, None] - x_train[None, :]
            k_star = sigma_f ** 2 * np.exp(-0.5 * diff_pred ** 2 / (length_scale ** 2))
            pred = k_star @ alpha_vec
            return np.where(np.isfinite(pred), pred, 0.0)
    
    # Solve L @ v = y_train
    v = np.linalg.solve(L, y_train)
    alpha_vec = np.linalg.solve(L.T, v)
    
    # k* = K(x_pred, x_train)
    diff_pred = X_pred[:, None] - x_train[None, :]
    k_star = sigma_f ** 2 * np.exp(-0.5 * diff_pred ** 2 / (length_scale ** 2))
    
    pred = k_star @ alpha_vec
    return np.where(np.isfinite(pred), pred, 0.0)


def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 285  # Safety margin for 330s limit

    X = np.asarray(X_train, dtype=float).reshape(-1, 1)
    y = np.asarray(y_train, dtype=float).flatten()
    n = len(y)

    if n < 2:
        mean_y = np.mean(y) if n > 0 else 0.0
        return lambda X: _safe_predict(lambda x: np.full(len(np.asarray(x).reshape(-1)), mean_y), X)

    x_vals = X[:, 0].copy()
    y_vals = y.copy()

    # Constant check
    if np.std(y_vals) < 1e-12:
        mean_y = np.mean(y_vals)
        return lambda X: _safe_predict(lambda x: np.full(len(np.asarray(x).reshape(-1)), mean_y), X)

    y_var = np.var(y_vals)
    if y_var < 1e-15:
        mean_y = np.mean(y_vals)
        return lambda X: _safe_predict(lambda x: np.full(len(np.asarray(x).reshape(-1)), mean_y), X)

    x_min, x_max = np.min(x_vals), np.max(x_vals)
    x_range = x_max - x_min
    if x_range < 1e-10:
        x_range = 1.0

    # Sort by x for better GP behavior
    sort_idx = np.argsort(x_vals)
    x_sorted = x_vals[sort_idx]
    y_sorted = y_vals[sort_idx]

    # Estimate initial parameters
    sigma_f_init = np.std(y_sorted)
    if sigma_f_init < 1e-12:
        sigma_f_init = 1.0
    
    # Length scale candidates: from very local to global
    ls_candidates = np.array([
        0.01 * x_range, 0.02 * x_range, 0.05 * x_range, 0.1 * x_range,
        0.2 * x_range, 0.3 * x_range, 0.5 * x_range, 0.7 * x_range,
        1.0 * x_range, 1.5 * x_range, 2.0 * x_range
    ])
    
    # Noise candidates (small since data is noiseless)
    noise_candidates = np.array([
        1e-12 * sigma_f_init, 1e-10 * sigma_f_init, 1e-8 * sigma_f_init,
        1e-6 * sigma_f_init, 1e-4 * sigma_f_init
    ])

    def neg_log_marginal_likelihood(log_params):
        """Negative log marginal likelihood for GP."""
        if time.time() > deadline - 2:
            return 1e10
        try:
            log_ls, log_sf, log_noise = log_params
            ls = np.exp(log_ls)
            sf = np.exp(log_sf)
            noise = np.exp(log_noise)
            
            if not (np.isfinite(ls) and np.isfinite(sf) and np.isfinite(noise)):
                return 1e10
            if ls < 1e-10 or sf < 1e-10 or noise < 1e-15:
                return 1e10
            
            # Kernel matrix
            diff = x_sorted[:, None] - x_sorted[None, :]
            K = sf ** 2 * np.exp(-0.5 * diff ** 2 / (ls ** 2))
            K += noise ** 2 * np.eye(n)
            
            # Add small jitter for numerical stability
            K += 1e-10 * np.eye(n)
            
            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                K += 1e-8 * np.eye(n)
                try:
                    L = np.linalg.cholesky(K)
                except np.linalg.LinAlgError:
                    return 1e10
            
            # Log determinant: 2 * sum(log(diag(L)))
            log_det = 2.0 * np.sum(np.log(np.diag(L)))
            
            # Solve L @ alpha = y
            try:
                alpha = np.linalg.solve(L, y_sorted)
            except np.linalg.LinAlgError:
                return 1e10
            
            # NLL = 0.5 * y^T K^{-1} y + 0.5 * log(det(K)) + n/2 * log(2*pi)
            nll = 0.5 * np.dot(y_sorted, alpha) + 0.5 * log_det + 0.5 * n * np.log(2 * np.pi)
            
            if not np.isfinite(nll):
                return 1e10
            
            return nll
        except Exception:
            return 1e10

    # Multi-start optimization over combinations of length scale and noise
    best_nll = np.inf
    best_params = None
    
    # Create grid of initial guesses
    ls_grid = np.log(ls_candidates)
    sf_grid = np.log([sigma_f_init * 0.5, sigma_f_init, sigma_f_init * 2.0])
    noise_grid = np.log(noise_candidates)
    
    # Limit number of starts to avoid timeout
    max_starts = 48
    start_count = 0
    
    for ls0 in ls_grid:
        if start_count >= max_starts:
            break
        for sf0 in sf_grid:
            if start_count >= max_starts:
                break
            for noise0 in noise_grid[:2]:  # Only first 2 noise levels to limit starts
                if start_count >= max_starts:
                    break
                
                x0 = np.array([ls0, sf0, noise0])
                
                try:
                    result = minimize(
                        neg_log_marginal_likelihood,
                        x0,
                        method='L-BFGS-B',
                        bounds=[(np.log(1e-6 * x_range), np.log(10 * x_range)),
                                (np.log(1e-6 * sigma_f_init), np.log(100 * sigma_f_init)),
                                (np.log(1e-14 * sigma_f_init), np.log(1e-2 * sigma_f_init))],
                        options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-12}
                    )
                    
                    if result.fun < best_nll and np.isfinite(result.fun):
                        best_nll = result.fun
                        best_params = result.x.copy()
                except Exception:
                    pass
                
                start_count += 1
    
    # If optimization failed, use best grid point
    if best_params is None:
        # Try a few fixed configurations
        best_nll = np.inf
        for ls in ls_candidates:
            for noise in noise_candidates[:2]:
                x0 = np.array([np.log(ls), np.log(sigma_f_init), np.log(noise)])
                nll = neg_log_marginal_likelihood(x0)
                if nll < best_nll and np.isfinite(nll):
                    best_nll = nll
                    best_params = x0.copy()
        
        if best_params is None:
            # Final fallback: use median length scale and small noise
            best_params = np.array([np.log(0.1 * x_range), np.log(sigma_f_init), np.log(1e-8 * sigma_f_init)])

    ls_opt = np.exp(best_params[0])
    sf_opt = np.exp(best_params[1])
    noise_opt = np.exp(best_params[2])

    # Validate: check training error
    try:
        train_pred = _gp_predict(x_sorted, x_sorted, y_sorted, ls_opt, sf_opt, noise_opt)
        train_mse = np.mean((train_pred - y_sorted) ** 2)
        
        if not np.isfinite(train_mse) or train_mse > 10 * y_var:
            # Fallback: try a different configuration
            ls_alt = 0.05 * x_range
            noise_alt = 1e-6 * sigma_f_init
            train_pred_alt = _gp_predict(x_sorted, x_sorted, y_sorted, ls_alt, sf_opt, noise_alt)
            train_mse_alt = np.mean((train_pred_alt - y_sorted) ** 2)
            if np.isfinite(train_mse_alt) and train_mse_alt < train_mse:
                ls_opt = ls_alt
                noise_opt = noise_alt
    except Exception:
        pass

    # Store optimal parameters
    final_ls = ls_opt
    final_sf = sf_opt
    final_noise = noise_opt

    # Polynomial fallback
    poly_fallback = None
    for deg in [2, 3, 4, 5]:
        if deg >= n:
            break
        try:
            pc = np.polyfit(x_vals, y_vals, deg)
            pp = np.polyval(pc, x_vals)
            if np.all(np.isfinite(pp)):
                poly_fallback = pc
                break
        except Exception:
            continue

    def predict(X_pred):
        X_pred = np.asarray(X_pred, dtype=float).reshape(-1, 1)[:, 0]
        n_pred = len(X_pred)
        try:
            pred = _gp_predict(X_pred, x_sorted, y_sorted, final_ls, final_sf, final_noise)
            if not np.all(np.isfinite(pred)):
                pred = np.where(np.isfinite(pred), pred, 0.0)
            return pred
        except Exception:
            if poly_fallback is not None:
                try:
                    return np.polyval(poly_fallback, X_pred)
                except Exception:
                    pass
            return np.zeros(n_pred)

    return predict
# EVOLVE-BLOCK-END
