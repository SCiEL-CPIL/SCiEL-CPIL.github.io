# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_P(support, weights):
    """Compute P[X1+X2+X3 < 2*X4] exactly in float64."""
    s = np.asarray(support, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    n = len(s)
    if n == 0:
        return 0.0
    
    w_sum = w.sum()
    if w_sum <= 1e-9:
        return 0.0
    w = w / w_sum
    
    # Sort by support value
    order = np.argsort(s)
    s_sorted = s[order]
    w_sorted = w[order]
    
    # We need: P = sum over k,l of w_k * w_l * (sum of w_i*w_j over pairs (i,j) with s_i+s_j < 2*s_l - s_k)
    
    # Precompute all pairwise sums and their probabilities
    # pair_sums[i,j] = s_i + s_j
    # pair_probs[i,j] = w_i * w_j
    pair_sums = s_sorted[:, None] + s_sorted[None, :]
    pair_probs = w_sorted[:, None] * w_sorted[None, :]
    
    # Flatten and sort pair sums for efficient lookup
    flat_sums = pair_sums.flatten()
    flat_probs = pair_probs.flatten()
    
    # Sort by sum value
    sort_idx = np.argsort(flat_sums)
    sorted_sums = flat_sums[sort_idx]
    sorted_probs = flat_probs[sort_idx]
    
    # Cumulative sum of probabilities for fast prefix queries
    cum_probs = np.cumsum(sorted_probs)
    
    # For each (k, l), we need sum of w_i*w_j where s_i+s_j < 2*s_l - s_k
    # This is a strict inequality, so we use searchsorted with 'left'
    
    P = 0.0
    for k in range(n):
        for l in range(n):
            threshold = 2.0 * s_sorted[l] - s_sorted[k]
            # Find number of pair sums strictly less than threshold
            idx = np.searchsorted(sorted_sums, threshold, side='left')
            if idx > 0:
                inner_sum = cum_probs[idx - 1]
            else:
                inner_sum = 0.0
            P += w_sorted[k] * w_sorted[l] * inner_sum
    
    return P

def _neg_P(params, n_points):
    """Negative P for minimization. params: first n_points are weights (log-space), rest are support points."""
    log_w = params[:n_points]
    support = params[n_points:]
    
    # Convert log weights to weights
    w = np.exp(log_w - np.max(log_w))  # For numerical stability
    
    return -_compute_P(support, w)

def _optimize_fixed_support(initial_support, initial_weights, max_iter=1000, timeout=300):
    """Optimize weights given fixed support points."""
    n = len(initial_support)
    
    def _neg_P_w(params):
        log_w = params
        w = np.exp(log_w - np.max(log_w))
        return -_compute_P(initial_support, w)
    
    # Initial log weights
    init_log_w = np.log(initial_weights + 1e-15)
    
    best_x = init_log_w.copy()
    best_val = _neg_P_w(best_x)
    
    start_time = time.time()
    
    # Try multiple optimization methods
    for method in ['Nelder-Mead', 'Powell']:
        if time.time() - start_time > timeout:
            break
        try:
            res = minimize(_neg_P_w, best_x, method=method, 
                         options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-10})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            pass
    
    # Also try L-BFGS-B with bounds
    if time.time() - start_time < timeout:
        try:
            bounds = [(-50, 50)] * n
            res = minimize(_neg_P_w, best_x, method='L-BFGS-B', bounds=bounds,
                         options={'maxiter': max_iter, 'ftol': 1e-12})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            pass
    
    w = np.exp(best_x - np.max(best_x))
    return initial_support, w, -best_val

def _optimize_support_and_weights(initial_support, initial_weights, max_iter=200, timeout=600):
    """Jointly optimize support and weights."""
    n = len(initial_support)
    
    def _neg_P_full(params):
        log_w = params[:n]
        support = params[n:]
        w = np.exp(log_w - np.max(log_w))
        # Ensure support is non-negative
        support = np.maximum(support, 0.0)
        return -_compute_P(support, w)
    
    init_log_w = np.log(initial_weights + 1e-15)
    init_params = np.concatenate([init_log_w, initial_support])
    
    best_params = init_params.copy()
    best_val = _neg_P_full(best_params)
    
    start_time = time.time()
    
    # Powell method for joint optimization
    try:
        res = minimize(_neg_P_full, best_params, method='Powell',
                     options={'maxiter': max_iter, 'xtol': 1e-6, 'ftol': 1e-10})
        if res.fun < best_val:
            best_val = res.fun
            best_params = res.x
    except Exception:
        pass
    
    if time.time() - start_time > timeout:
        log_w = best_params[:n]
        support = np.maximum(best_params[n:], 0.0)
        w = np.exp(log_w - np.max(log_w))
        return support, w, -best_val
    
    # Nelder-Mead
    try:
        res = minimize(_neg_P_full, best_params, method='Nelder-Mead',
                     options={'maxiter': max_iter, 'xatol': 1e-6, 'fatol': 1e-10})
        if res.fun < best_val:
            best_val = res.fun
            best_params = res.x
    except Exception:
        pass
    
    log_w = best_params[:n]
    support = np.maximum(best_params[n:], 0.0)
    w = np.exp(log_w - np.max(log_w))
    return support, w, -best_val

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.
    
    Returns:
        support: np.ndarray of shape (n,), non-negative support points of the measure.
        weights: np.ndarray of shape (n,), non-negative weights (need not be normalized).
    """
    deadline = time.time() + 620  # Safety margin before 630s kill
    
    best_support = None
    best_weights = None
    best_P = -1.0
    
    # Start with a reasonable initial distribution
    # Try different initial structures
    
    initial_configs = []
    
    # Config 1: Uniform on {0, 1, ..., n-1} with n=8
    n1 = 8
    support1 = np.arange(n1, dtype=np.float64)
    weights1 = np.ones(n1, dtype=np.float64) / n1
    initial_configs.append((support1, weights1))
    
    # Config 2: Skewed distribution - more mass at low values
    n2 = 8
    support2 = np.arange(n2, dtype=np.float64)
    weights2 = np.array([0.3, 0.2, 0.15, 0.1, 0.08, 0.06, 0.05, 0.06], dtype=np.float64)
    weights2 = weights2 / weights2.sum()
    initial_configs.append((support2, weights2))
    
    # Config 3: Bimodal - mass at 0 and at higher values
    n3 = 6
    support3 = np.array([0, 1, 2, 4, 6, 8], dtype=np.float64)
    weights3 = np.array([0.25, 0.15, 0.1, 0.2, 0.15, 0.15], dtype=np.float64)
    weights3 = weights3 / weights3.sum()
    initial_configs.append((support3, weights3))
    
    # Config 4: Exponential-like decay
    n4 = 10
    support4 = np.arange(n4, dtype=np.float64)
    weights4 = np.exp(-0.3 * support4)
    weights4 = weights4 / weights4.sum()
    initial_configs.append((support4, weights4))
    
    # Config 5: Mass concentrated at 0 and 1
    n5 = 4
    support5 = np.array([0, 1, 2, 4], dtype=np.float64)
    weights5 = np.array([0.4, 0.3, 0.2, 0.1], dtype=np.float64)
    weights5 = weights5 / weights5.sum()
    initial_configs.append((support5, weights5))
    
    # Evaluate initial configs
    for support, weights in initial_configs:
        P = _compute_P(support, weights)
        if P > best_P:
            best_P = P
            best_support = support.copy()
            best_weights = weights.copy()
    
    if time.time() > deadline:
        return best_support, best_weights
    
    # Optimize each config
    for support, weights in initial_configs:
        if time.time() > deadline - 60:
            break
        
        # First optimize weights with fixed support
        try:
            opt_support, opt_weights, P = _optimize_fixed_support(support, weights, max_iter=500, timeout=100)
            if P > best_P:
                best_P = P
                best_support = opt_support.copy()
                best_weights = opt_weights.copy()
        except Exception:
            pass
        
        if time.time() > deadline - 30:
            break
        
        # Then jointly optimize support and weights
        try:
            opt_support, opt_weights, P = _optimize_support_and_weights(opt_support, opt_weights, max_iter=100, timeout=120)
            if P > best_P:
                best_P = P
                best_support = opt_support.copy()
                best_weights = opt_weights.copy()
        except Exception:
            pass
    
    if time.time() > deadline:
        return best_support, best_weights
    
    # Try additional random restarts with the best structure as starting point
    if best_support is not None:
        n = len(best_support)
        for _ in range(3):
            if time.time() > deadline - 20:
                break
            
            # Perturb the best solution
            np.random.seed(42 + int(time.time() * 1000) % 1000)
            perturbed_support = best_support + np.random.randn(n) * 0.1
            perturbed_support = np.maximum(perturbed_support, 0.0)
            perturbed_weights = best_weights * (1 + np.random.randn(n) * 0.1)
            perturbed_weights = np.maximum(perturbed_weights, 1e-6)
            perturbed_weights = perturbed_weights / perturbed_weights.sum()
            
            try:
                opt_support, opt_weights, P = _optimize_support_and_weights(
                    perturbed_support, perturbed_weights, max_iter=100, timeout=60)
                if P > best_P:
                    best_P = P
                    best_support = opt_support.copy()
                    best_weights = opt_weights.copy()
            except Exception:
                pass
    
    # Ensure validity
    if best_support is None:
        best_support = np.array([0.0, 1.0], dtype=np.float64)
        best_weights = np.array([0.5, 0.5], dtype=np.float64)
    
    # Clean up: ensure no NaN/inf, non-negative
    best_support = np.nan_to_num(best_support, nan=0.0, posinf=1e6, neginf=0.0)
    best_weights = np.nan_to_num(best_weights, nan=0.0, posinf=1.0, neginf=0.0)
    best_support = np.maximum(best_support, 0.0)
    best_weights = np.maximum(best_weights, 0.0)
    
    # Ensure sum of weights is positive
    if best_weights.sum() <= 1e-9:
        best_weights = np.ones_like(best_weights)
    
    return best_support, best_weights

# EVOLVE-BLOCK-END
