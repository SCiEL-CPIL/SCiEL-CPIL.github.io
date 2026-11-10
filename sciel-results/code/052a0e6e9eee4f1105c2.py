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
    sort_idx = np.argsort(flat_sums, kind='stable')
    sorted_sums = flat_sums[sort_idx]
    sorted_probs = flat_probs[sort_idx]
    
    # Cumulative sum of probabilities for fast prefix queries
    cum_probs = np.cumsum(sorted_probs)
    
    # Vectorized computation over (k, l) pairs
    thresholds = 2.0 * s_sorted[None, :] - s_sorted[:, None]
    
    # For each threshold, find count of pair sums strictly less
    counts = np.searchsorted(sorted_sums, thresholds.ravel(), side='left')
    counts = counts.reshape(n, n)
    
    safe_counts = np.maximum(counts - 1, 0)
    inner_sums = np.where(counts > 0, cum_probs[safe_counts], 0.0)
    
    P = np.sum(w_sorted[:, None] * w_sorted[None, :] * inner_sums)
    
    return float(P)

def _precompute_support(support):
    """Precompute sorted pair sums for a fixed support."""
    s = np.asarray(support, dtype=np.float64)
    order = np.argsort(s)
    s_sorted = s[order]
    pair_sums = s_sorted[:, None] + s_sorted[None, :]
    flat_sums = pair_sums.flatten()
    sort_order = np.argsort(flat_sums, kind='stable')
    sorted_sums = flat_sums[sort_order]
    return s_sorted, sorted_sums, sort_order

def _compute_P_fixed_support(s_sorted, weights, sorted_sums, sort_order):
    """Fast P computation when support is fixed and sorted_sums precomputed."""
    w = np.asarray(weights, dtype=np.float64)
    n = len(s_sorted)
    
    w_sum = w.sum()
    if w_sum <= 1e-9:
        return 0.0
    w = w / w_sum
    
    pair_probs = w[:, None] * w[None, :]
    flat_probs = pair_probs.flatten()
    sorted_probs = flat_probs[sort_order]
    cum_probs = np.cumsum(sorted_probs)
    
    thresholds = 2.0 * s_sorted[None, :] - s_sorted[:, None]
    counts = np.searchsorted(sorted_sums, thresholds.ravel(), side='left')
    counts = counts.reshape(n, n)
    
    safe_counts = np.maximum(counts - 1, 0)
    inner_sums = np.where(counts > 0, cum_probs[safe_counts], 0.0)
    
    P = np.sum(w[:, None] * w[None, :] * inner_sums)
    return float(P)

def _neg_P_w(params, s_sorted, sorted_sums, sort_order):
    """Negative P for weight optimization with fixed support."""
    log_w = params
    w = np.exp(log_w - np.max(log_w))
    return -_compute_P_fixed_support(s_sorted, w, sorted_sums, sort_order)

def _optimize_fixed_support(initial_support, initial_weights, max_iter=500, timeout=120):
    """Optimize weights given fixed support points using precomputed structure."""
    n = len(initial_support)
    s_sorted, sorted_sums, sort_order = _precompute_support(initial_support)
    
    init_log_w = np.log(initial_weights + 1e-15)
    best_x = init_log_w.copy()
    best_val = _neg_P_w(best_x, s_sorted, sorted_sums, sort_order)
    
    start_time = time.time()
    
    try:
        res = minimize(_neg_P_w, best_x, args=(s_sorted, sorted_sums, sort_order),
                     method='Powell', options={'maxiter': max_iter, 'xtol': 1e-10, 'ftol': 1e-12})
        if res.fun < best_val:
            best_val = res.fun
            best_x = res.x
    except Exception:
        pass
    
    if time.time() - start_time < timeout:
        try:
            res = minimize(_neg_P_w, best_x, args=(s_sorted, sorted_sums, sort_order),
                         method='Nelder-Mead', options={'maxiter': max_iter, 'xatol': 1e-10, 'fatol': 1e-12})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            pass
    
    if time.time() - start_time < timeout:
        try:
            bounds = [(-30, 30)] * n
            res = minimize(_neg_P_w, best_x, args=(s_sorted, sorted_sums, sort_order),
                         method='L-BFGS-B', bounds=bounds, options={'maxiter': max_iter, 'ftol': 1e-14})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            pass
    
    w = np.exp(best_x - np.max(best_x))
    return initial_support, w, -best_val

def _neg_P_full(params, n_points):
    """Negative P for joint optimization."""
    log_w = params[:n_points]
    support = params[n_points:]
    w = np.exp(log_w - np.max(log_w))
    support = np.maximum(support, 0.0)
    support = np.sort(support)
    return -_compute_P(support, w)

def _optimize_support_and_weights(initial_support, initial_weights, max_iter=300, timeout=150):
    """Jointly optimize support and weights."""
    n = len(initial_support)
    
    init_log_w = np.log(initial_weights + 1e-15)
    init_support = np.sort(initial_support)
    init_params = np.concatenate([init_log_w, init_support])
    
    best_params = init_params.copy()
    best_val = _neg_P_full(best_params, n)
    
    start_time = time.time()
    
    try:
        res = minimize(_neg_P_full, best_params, args=(n,), method='Powell',
                     options={'maxiter': max_iter, 'xtol': 1e-8, 'ftol': 1e-12})
        if res.fun < best_val:
            best_val = res.fun
            best_params = res.x
    except Exception:
        pass
    
    if time.time() - start_time < timeout:
        try:
            res = minimize(_neg_P_full, best_params, args=(n,), method='Nelder-Mead',
                         options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-12})
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
    deadline = time.time() + 580  # Safety margin
    
    best_support = None
    best_weights = None
    best_P = -1.0
    
    def _update_best(s, w, P):
        nonlocal best_support, best_weights, best_P
        if P > best_P:
            best_P = P
            best_support = s.copy()
            best_weights = w.copy()
    
    # Generate diverse initial configurations
    initial_configs = []
    
    for n in [3, 4, 5, 6, 8, 10, 12]:
        # Uniform
        s = np.linspace(0, 1, n, dtype=np.float64)
        w = np.ones(n, dtype=np.float64) / n
        initial_configs.append((s, w))
        
        # Heavy at 0
        s = np.linspace(0, 1, n, dtype=np.float64)
        w = np.exp(-3.0 * s)
        w = w / w.sum()
        initial_configs.append((s, w))
        
        # Very heavy at 0
        s = np.linspace(0, 1, n, dtype=np.float64)
        w = np.exp(-5.0 * s)
        w = w / w.sum()
        initial_configs.append((s, w))
        
        # Bimodal
        if n >= 4:
            half = n // 2
            s = np.concatenate([np.linspace(0, 0.2, half), np.linspace(0.5, 1.0, n - half)])
            w = np.ones(n, dtype=np.float64) / n
            initial_configs.append((s, w))
            
            # Bimodal with weight bias
            s = np.concatenate([np.linspace(0, 0.2, half), np.linspace(0.5, 1.0, n - half)])
            w = np.concatenate([np.ones(half) * 0.6, np.ones(n - half) * 0.4])
            w = w / w.sum()
            initial_configs.append((s, w))
        
        # Geometric spacing
        s = np.array([0.0] + [0.5 ** (i) for i in range(1, n)])
        s = s / s.max()
        w = np.ones(n, dtype=np.float64) / n
        initial_configs.append((s, w))
    
    # Specific structured configs
    s = np.array([0.0, 0.05, 0.1, 0.3, 0.7, 1.0])
    w = np.array([0.35, 0.2, 0.15, 0.1, 0.1, 0.1])
    initial_configs.append((s, w))
    
    s = np.array([0.0, 0.1, 0.5, 1.0])
    w = np.array([0.5, 0.2, 0.15, 0.15])
    initial_configs.append((s, w))
    
    s = np.array([0.0, 0.0, 0.5, 1.0])
    w = np.array([0.3, 0.2, 0.25, 0.25])
    initial_configs.append((s, w))
    
    s = np.array([0.0, 0.2, 0.6, 1.0])
    w = np.array([0.4, 0.25, 0.2, 0.15])
    initial_configs.append((s, w))
    
    # Evaluate all initial configs
    for support, weights in initial_configs:
        P = _compute_P(support, weights)
        _update_best(support, weights, P)
    
    if time.time() > deadline:
        return best_support, best_weights
    
    # Sort configs by initial P
    config_scores = []
    for i, (support, weights) in enumerate(initial_configs):
        P = _compute_P(support, weights)
        config_scores.append((P, i, support, weights))
    config_scores.sort(key=lambda x: -x[0])
    
    # Optimize top configs
    num_to_optimize = min(8, len(config_scores))
    for P_init, idx, support, weights in config_scores[:num_to_optimize]:
        if time.time() > deadline - 80:
            break
        
        # Optimize weights with fixed support (faster)
        try:
            opt_support, opt_weights, P = _optimize_fixed_support(support, weights, max_iter=400, timeout=60)
            _update_best(opt_support, opt_weights, P)
        except Exception:
            pass
        
        if time.time() > deadline - 40:
            break
        
        # Joint optimization
        try:
            opt_support, opt_weights, P = _optimize_support_and_weights(opt_support, opt_weights, max_iter=200, timeout=80)
            _update_best(opt_support, opt_weights, P)
        except Exception:
            pass
    
    if time.time() > deadline:
        return best_support, best_weights
    
    # Final polish: high-precision local optimization on best solution
    if best_support is not None and time.time() < deadline - 30:
        n = len(best_support)
        
        for trial in range(8):
            if time.time() > deadline - 15:
                break
            
            rng = np.random.RandomState(42 + trial * 137)
            scale = 0.03 * (1 + trial * 0.5)
            perturbed_support = best_support + rng.randn(n) * scale
            perturbed_support = np.maximum(perturbed_support, 0.0)
            perturbed_support = np.sort(perturbed_support)
            
            perturbed_weights = best_weights * (1 + rng.randn(n) * 0.05)
            perturbed_weights = np.maximum(perturbed_weights, 1e-8)
            perturbed_weights = perturbed_weights / perturbed_weights.sum()
            
            try:
                opt_support, opt_weights, P = _optimize_fixed_support(
                    perturbed_support, perturbed_weights, max_iter=300, timeout=25)
                _update_best(opt_support, opt_weights, P)
            except Exception:
                pass
            
            if time.time() > deadline - 10:
                break
            
            try:
                opt_support, opt_weights, P = _optimize_support_and_weights(
                    opt_support, opt_weights, max_iter=150, timeout=20)
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
