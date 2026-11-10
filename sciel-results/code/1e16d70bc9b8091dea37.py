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
    
    # Precompute all pairwise sums and their probabilities
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
    
    # Vectorized computation over (k, l) pairs
    # thresholds[k, l] = 2*s_l - s_k
    thresholds = 2.0 * s_sorted[None, :] - s_sorted[:, None]
    
    # For each threshold, find count of pair sums strictly less
    # searchsorted with side='left' gives index of first element >= threshold
    # So count of elements < threshold is that index
    counts = np.searchsorted(sorted_sums, thresholds.ravel(), side='left')
    counts = counts.reshape(n, n)
    
    # Get cumulative probabilities at those indices (handle index 0)
    # If count is 0, inner_sum is 0; otherwise cum_probs[count-1]
    safe_counts = np.maximum(counts - 1, 0)
    inner_sums = np.where(counts > 0, cum_probs[safe_counts], 0.0)
    
    # P = sum over k,l of w_k * w_l * inner_sum[k,l]
    P = np.sum(w_sorted[:, None] * w_sorted[None, :] * inner_sums)
    
    return float(P)

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
        # Ensure support is non-negative and sorted
        support = np.maximum(support, 0.0)
        support = np.sort(support)
        return -_compute_P(support, w)
    
    init_log_w = np.log(initial_weights + 1e-15)
    init_support = np.sort(initial_support)
    init_params = np.concatenate([init_log_w, init_support])
    
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
        support = np.sort(support)
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
    support = np.sort(support)
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
    deadline = time.time() + 600  # Safety margin before 630s kill
    
    best_support = None
    best_weights = None
    best_P = -1.0
    
    def _update_best(s, w, P):
        nonlocal best_support, best_weights, best_P
        if P > best_P:
            best_P = P
            best_support = s.copy()
            best_weights = w.copy()
    
    # Generate diverse initial configurations across different sizes
    initial_configs = []
    
    # n=3: Simple configurations
    for n in [3, 4, 5, 6, 8, 10]:
        # Uniform
        s = np.linspace(0, 1, n, dtype=np.float64)
        w = np.ones(n, dtype=np.float64) / n
        initial_configs.append((s, w))
        
        # Skewed toward 0
        s = np.linspace(0, 1, n, dtype=np.float64)
        w = np.exp(-2.0 * s)
        w = w / w.sum()
        initial_configs.append((s, w))
        
        # Bimodal
        if n >= 4:
            s = np.concatenate([np.linspace(0, 0.3, n//2), np.linspace(0.7, 1.0, n - n//2)])
            w = np.ones(n, dtype=np.float64) / n
            initial_configs.append((s, w))
        
        # Concentrated at 0
        s = np.array([0.0] + list(np.linspace(0.2, 1.0, n-1)))
        w = np.ones(n, dtype=np.float64) / n
        initial_configs.append((s, w))
    
    # Special structured configurations
    # Heavy mass at 0
    s = np.array([0.0, 0.1, 0.5, 1.0], dtype=np.float64)
    w = np.array([0.5, 0.2, 0.2, 0.1], dtype=np.float64)
    initial_configs.append((s, w))
    
    # Three-point with specific structure
    s = np.array([0.0, 0.4, 1.0], dtype=np.float64)
    w = np.array([0.4, 0.3, 0.3], dtype=np.float64)
    initial_configs.append((s, w))
    
    # Evaluate all initial configs
    for support, weights in initial_configs:
        P = _compute_P(support, weights)
        _update_best(support, weights, P)
    
    if time.time() > deadline:
        return best_support, best_weights
    
    # Optimize the top configurations
    # Sort configs by initial P and try best ones first
    config_scores = []
    for i, (support, weights) in enumerate(initial_configs):
        P = _compute_P(support, weights)
        config_scores.append((P, i, support, weights))
    config_scores.sort(key=lambda x: -x[0])
    
    # Optimize top 6 configs
    for P_init, idx, support, weights in config_scores[:6]:
        if time.time() > deadline - 60:
            break
        
        # First optimize weights with fixed support
        try:
            opt_support, opt_weights, P = _optimize_fixed_support(support, weights, max_iter=300, timeout=80)
            _update_best(opt_support, opt_weights, P)
        except Exception:
            pass
        
        if time.time() > deadline - 30:
            break
        
        # Then jointly optimize support and weights
        try:
            opt_support, opt_weights, P = _optimize_support_and_weights(opt_support, opt_weights, max_iter=150, timeout=100)
            _update_best(opt_support, opt_weights, P)
        except Exception:
            pass
    
    if time.time() > deadline:
        return best_support, best_weights
    
    # Additional local polish on the best solution
    if best_support is not None and time.time() < deadline - 20:
        n = len(best_support)
        
        # Try small perturbations with local optimization
        for trial in range(5):
            if time.time() > deadline - 10:
                break
            
            rng = np.random.RandomState(42 + trial * 137)
            scale = 0.05 * (trial + 1)
            perturbed_support = best_support + rng.randn(n) * scale
            perturbed_support = np.maximum(perturbed_support, 0.0)
            # Ensure sorted
            perturbed_support = np.sort(perturbed_support)
            
            perturbed_weights = best_weights * (1 + rng.randn(n) * 0.1)
            perturbed_weights = np.maximum(perturbed_weights, 1e-6)
            perturbed_weights = perturbed_weights / perturbed_weights.sum()
            
            try:
                opt_support, opt_weights, P = _optimize_support_and_weights(
                    perturbed_support, perturbed_weights, max_iter=100, timeout=40)
                _update_best(opt_support, opt_weights, P)
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
