# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _compute_P(support, weights):
    """Compute P[X1+X2+X3 < 2*X4] for given support and weights."""
    w = weights / weights.sum()
    n = len(support)
    s = support
    
    # Precompute pairwise sums s_i + s_j
    # P = sum_{k,l} w_k * w_l * (sum_{i,j: s_i+s_j < 2*s_l - s_k} w_i * w_j)
    
    # For efficiency, for each (k, l) pair, we need the probability that X1+X2 < 2*s_l - s_k
    # This is a 2D convolution-like computation
    
    # Build all pairwise sums
    pair_sums = s[:, None] + s[None, :]  # n x n
    pair_weights = w[:, None] * w[None, :]  # n x n
    
    P = 0.0
    for k in range(n):
        for l in range(n):
            threshold = 2.0 * s[l] - s[k]
            if threshold <= 0:
                continue
            # Sum pair_weights where pair_sums < threshold
            mask = pair_sums < threshold
            P += w[k] * w[l] * np.sum(pair_weights[mask])
    
    return P

def _objective(params, support_template):
    """Objective function for optimization: negative P."""
    n = len(support_template)
    # params: log-weights (n-1 of them, last determined by normalization)
    log_w = params[:n]
    w = np.exp(log_w - log_w.max())  # stable softmax-like
    w = w / w.sum()
    
    # Support points are fixed by template
    return -_compute_P(support_template, w)

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.
    """
    start_time = time.time()
    deadline = start_time + 600.0  # 600s limit, keep safety margin
    
    # Strategy: Use a geometric progression support with optimized weights
    # The key insight: put mass at 0 and at large values to exploit the asymmetry
    
    # Try different support structures
    best_P = 0.0
    best_support = None
    best_weights = None
    
    # Template 1: Geometric progression {0, 1, r, r^2, ..., r^(n-1)}
    # Template 2: {0, 1, 2, ..., n-1} (uniform, baseline)
    # Template 3: Bimodal - many small values, few large values
    
    # Start with a reasonable geometric progression
    n_points = 16
    r = 2.0  # geometric ratio
    support = np.array([0.0] + [r**i for i in range(1, n_points)])
    
    # Initial weights: heavy at 0, decreasing
    log_w_init = np.array([0.0] + np.linspace(0, -3, n_points-1))
    
    # Optimize weights for this support
    result = minimize(
        _objective,
        log_w_init,
        args=(support,),
        method='L-BFGS-B',
        options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
    )
    
    if result.success or result.fun < 0:
        log_w_opt = result.x
        w_opt = np.exp(log_w_opt - log_w_opt.max())
        w_opt = w_opt / w_opt.sum()
        P_val = -result.fun
        if P_val > best_P:
            best_P = P_val
            best_support = support.copy()
            best_weights = w_opt.copy()
    
    elapsed = time.time() - start_time
    if elapsed > 500:  # Check time
        if best_support is not None:
            return best_support, best_weights
        # Fallback to baseline
        n = 32
        support = np.arange(n, dtype=np.float64)
        weights = np.ones(n, dtype=np.float64) / n
        return support, weights
    
    # Try different geometric ratios
    for r_test in [1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
        if time.time() > deadline - 100:
            break
        n_points = 14
        support_test = np.array([0.0] + [r_test**i for i in range(1, n_points)])
        
        # Initial weights: heavy at 0
        log_w_init = np.array([0.0] + np.linspace(0, -4, n_points-1))
        
        result = minimize(
            _objective,
            log_w_init,
            args=(support_test,),
            method='L-BFGS-B',
            options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        if result.success or result.fun < 0:
            log_w_opt = result.x
            w_opt = np.exp(log_w_opt - log_w_opt.max())
            w_opt = w_opt / w_opt.sum()
            P_val = -result.fun
            if P_val > best_P:
                best_P = P_val
                best_support = support_test.copy()
                best_weights = w_opt.copy()
    
    # Try a mixed structure: {0, 1, 2, 3, L, 2L, 3L} for various L
    for L in [10, 20, 50, 100, 200]:
        if time.time() > deadline - 100:
            break
        support_test = np.array([0.0, 1.0, 2.0, 3.0, float(L), 2.0*L, 3.0*L])
        n_pts = len(support_test)
        
        # Initial: heavy at 0 and 1
        log_w_init = np.array([0.0, 0.0, -1.0, -2.0, -3.0, -4.0, -5.0])
        
        result = minimize(
            _objective,
            log_w_init,
            args=(support_test,),
            method='L-BFGS-B',
            options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        if result.success or result.fun < 0:
            log_w_opt = result.x
            w_opt = np.exp(log_w_opt - log_w_opt.max())
            w_opt = w_opt / w_opt.sum()
            P_val = -result.fun
            if P_val > best_P:
                best_P = P_val
                best_support = support_test.copy()
                best_weights = w_opt.copy()
    
    # Try the known good structure: {0, 1, 3} with optimized weights
    support_test = np.array([0.0, 1.0, 3.0])
    log_w_init = np.array([0.0, 0.0, -1.0])
    result = minimize(
        _objective,
        log_w_init,
        args=(support_test,),
        method='L-BFGS-B',
        options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10}
    )
    if result.success or result.fun < 0:
        log_w_opt = result.x
        w_opt = np.exp(log_w_opt - log_w_opt.max())
        w_opt = w_opt / w_opt.sum()
        P_val = -result.fun
        if P_val > best_P:
            best_P = P_val
            best_support = support_test.copy()
            best_weights = w_opt.copy()
    
    # Try {0, 1, 4}
    support_test = np.array([0.0, 1.0, 4.0])
    log_w_init = np.array([0.0, 0.0, -1.0])
    result = minimize(
        _objective,
        log_w_init,
        args=(support_test,),
        method='L-BFGS-B',
        options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10}
    )
    if result.success or result.fun < 0:
        log_w_opt = result.x
        w_opt = np.exp(log_w_opt - log_w_opt.max())
        w_opt = w_opt / w_opt.sum()
        P_val = -result.fun
        if P_val > best_P:
            best_P = P_val
            best_support = support_test.copy()
            best_weights = w_opt.copy()
    
    # Try {0, 1, 5}
    support_test = np.array([0.0, 1.0, 5.0])
    log_w_init = np.array([0.0, 0.0, -1.0])
    result = minimize(
        _objective,
        log_w_init,
        args=(support_test,),
        method='L-BFGS-B',
        options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10}
    )
    if result.success or result.fun < 0:
        log_w_opt = result.x
        w_opt = np.exp(log_w_opt - log_w_opt.max())
        w_opt = w_opt / w_opt.sum()
        P_val = -result.fun
        if P_val > best_P:
            best_P = P_val
            best_support = support_test.copy()
            best_weights = w_opt.copy()
    
    # Try {0, 1, 2, 6}
    support_test = np.array([0.0, 1.0, 2.0, 6.0])
    log_w_init = np.array([0.0, 0.0, -1.0, -2.0])
    result = minimize(
        _objective,
        log_w_init,
        args=(support_test,),
        method='L-BFGS-B',
        options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10}
    )
    if result.success or result.fun < 0:
        log_w_opt = result.x
        w_opt = np.exp(log_w_opt - log_w_opt.max())
        w_opt = w_opt / w_opt.sum()
        P_val = -result.fun
        if P_val > best_P:
            best_P = P_val
            best_support = support_test.copy()
            best_weights = w_opt.copy()
    
    # Final validation
    if best_support is None:
        # Fallback to baseline
        n = 32
        support = np.arange(n, dtype=np.float64)
        weights = np.ones(n, dtype=np.float64) / n
        return support, weights
    
    # Ensure validity
    support = np.asarray(best_support, dtype=np.float64).flatten()
    weights = np.asarray(best_weights, dtype=np.float64).flatten()
    
    # Clip any tiny negative values from numerical error
    weights = np.maximum(weights, 0.0)
    support = np.maximum(support, 0.0)
    
    # Ensure sum of weights > 1e-9
    w_sum = weights.sum()
    if w_sum <= 1e-9:
        weights = np.ones_like(weights) / len(weights)
    else:
        weights = weights / w_sum
    
    # Ensure no NaN or inf
    if not np.all(np.isfinite(support)) or not np.all(np.isfinite(weights)):
        n = 32
        support = np.arange(n, dtype=np.float64)
        weights = np.ones(n, dtype=np.float64) / n
    
    return support, weights

# EVOLVE-BLOCK-END
