# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_P(support, weights):
    """
    Compute P[X1+X2+X3 < 2*X4] for discrete measure with given support and weights.
    Uses O(n^2 log n) vectorized approach.
    """
    s = np.asarray(support, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    n = len(s)
    if n == 0:
        return 0.0
    w = w / w.sum()
    
    # For each l (X4) and k (X1), compute P[X2+X3 < 2*s[l] - s[k]]
    # P = sum_k sum_l w[k]*w[l] * P[X2+X3 < 2*s[l]-s[k]]
    
    # Compute all pairwise sums s[i]+s[j] and their probabilities w[i]*w[j]
    # Then for each target t, find pairs with sum < t
    
    # Vectorized: for each (k, l) pair, target = 2*s[l] - s[k]
    # We need P[X2+X3 < target] = sum_{i,j: s[i]+s[j] < target} w[i]*w[j]
    
    # Precompute pairwise sums
    # s_pair = s[:, None] + s[None, :]  shape (n, n)
    # w_pair = w[:, None] * w[None, :]  shape (n, n)
    
    # For efficiency, sort the pairwise sums and use cumulative sums
    # But targets depend on (k, l)
    
    # Alternative: for each l, for each k, binary search
    # This is O(n^2 * log(n^2)) = O(n^2 * log n)
    
    s_pair = s[:, None] + s[None, :]  # (n, n)
    w_pair = w[:, None] * w[None, :]  # (n, n)
    
    # Flatten and sort
    flat_s = s_pair.ravel()  # (n*n,)
    flat_w = w_pair.ravel()  # (n*n,)
    
    # Sort by s_pair
    order = np.argsort(flat_s, kind='mergesort')
    sorted_s = flat_s[order]
    sorted_w = flat_w[order]
    cum_w = np.cumsum(sorted_w)
    
    # For each (k, l), target = 2*s[l] - s[k]
    # Find number of pairs with sum < target using searchsorted
    # targets shape: (n, n) where targets[k, l] = 2*s[l] - s[k]
    targets = 2.0 * s[None, :] - s[:, None]  # (n, n), targets[k, l]
    
    # For each target, find the index where sorted_s >= target
    # Number of pairs with sum < target = searchsorted(sorted_s, target, side='left')
    # Then P[X2+X3 < target] = cum_w[idx-1] if idx > 0 else 0
    
    # Vectorized searchsorted
    flat_targets = targets.ravel()  # (n*n,)
    idx = np.searchsorted(sorted_s, flat_targets, side='left')  # (n*n,)
    
    # Get cumulative sum up to idx-1
    # For idx=0, result is 0
    # For idx>0, result is cum_w[idx-1]
    P_X2X3 = np.zeros_like(flat_targets)
    mask = idx > 0
    P_X2X3[mask] = cum_w[idx[mask] - 1]
    
    # Reshape to (n, n)
    P_X2X3 = P_X2X3.reshape(n, n)
    
    # P = sum_k sum_l w[k]*w[l] * P_X2X3[k, l]
    P = np.sum(w[:, None] * w[None, :] * P_X2X3)
    
    return P

def _neg_P(params, support_fixed=None):
    """Negative P for minimization. params are weights only (support fixed)."""
    w = np.abs(params)
    return -_compute_P(support_fixed, w)

def _optimize_weights(support, initial_weights, time_budget=10.0):
    """Optimize weights for fixed support using L-BFGS-B."""
    start_time = time.time()
    n = len(support)
    
    # Normalize initial weights
    w0 = np.asarray(initial_weights, dtype=np.float64)
    w0 = np.abs(w0)
    if w0.sum() <= 1e-12:
        w0 = np.ones(n) / n
    else:
        w0 = w0 / w0.sum()
    
    best_P = _compute_P(support, w0)
    best_w = w0.copy()
    
    # Use L-BFGS-B with bounds
    # We optimize in log-space or directly with constraints
    # Direct approach: minimize -P(w) subject to w >= 0, sum(w) = 1
    # Use change of variables: w = exp(x) / sum(exp(x))
    
    def _log_w_to_w(x):
        x_shifted = x - x.max()  # for numerical stability
        e = np.exp(x_shifted)
        return e / e.sum()
    
    def _neg_P_log(x):
        w = _log_w_to_w(x)
        return -_compute_P(support, w)
    
    # Initial x
    x0 = np.log(w0 + 1e-300)
    
    # Try L-BFGS-B
    try:
        result = minimize(_neg_P_log, x0, method='L-BFGS-B', 
                         options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
        
        if result.success or result.fun < -best_P:
            w_opt = _log_w_to_w(result.x)
            P_opt = _compute_P(support, w_opt)
            if P_opt > best_P:
                best_P = P_opt
                best_w = w_opt.copy()
    except Exception:
        pass
    
    # Also try Nelder-Mead on normalized weights
    if time.time() - start_time < time_budget:
        try:
            def _neg_P_nm(w):
                w = np.abs(w)
                s = w.sum()
                if s <= 1e-12:
                    return 0.0
                return -_compute_P(support, w / s)
            
            result2 = minimize(_neg_P_nm, w0, method='Nelder-Mead',
                              options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-12})
            if result2.fun < -best_P:
                w_opt = np.abs(result2.x)
                s = w_opt.sum()
                if s > 1e-12:
                    w_opt = w_opt / s
                    P_opt = _compute_P(support, w_opt)
                    if P_opt > best_P:
                        best_P = P_opt
                        best_w = w_opt.copy()
        except Exception:
            pass
    
    return best_w, best_P

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.
    """
    start_time = time.time()
    deadline = start_time + 620.0  # 10 second safety margin
    
    best_support = None
    best_weights = None
    best_P = -1.0
    
    # Try various support sizes and configurations
    # Start with small n for quick exploration, then try larger n if time permits
    
    # Configuration 1: Uniform on {0, 1, ..., n-1}
    for n in [2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32]:
        if time.time() > deadline:
            break
        
        support = np.arange(n, dtype=np.float64)
        weights = np.ones(n, dtype=np.float64) / n
        
        P = _compute_P(support, weights)
        if P > best_P:
            best_P = P
            best_support = support.copy()
            best_weights = weights.copy()
        
        # Optimize weights for this support
        if time.time() > deadline:
            break
        
        time_budget = min(15.0, max(2.0, (deadline - time.time()) * 0.3))
        w_opt, P_opt = _optimize_weights(support, weights, time_budget=time_budget)
        
        if P_opt > best_P:
            best_P = P_opt
            best_support = support.copy()
            best_weights = w_opt.copy()
    
    # Try geometric/exponential supports
    for n in [3, 4, 5, 6, 8]:
        if time.time() > deadline:
            break
        
        # Exponential support: s_i = c * i
        for c in [0.5, 1.0, 2.0, 5.0]:
            if time.time() > deadline:
                break
            
            support = c * np.arange(n, dtype=np.float64)
            weights = np.ones(n, dtype=np.float64) / n
            
            P = _compute_P(support, weights)
            if P > best_P:
                best_P = P
                best_support = support.copy()
                best_weights = weights.copy()
            
            if time.time() > deadline:
                break
            
            time_budget = min(10.0, max(1.0, (deadline - time.time()) * 0.2))
            w_opt, P_opt = _optimize_weights(support, weights, time_budget=time_budget)
            
            if P_opt > best_P:
                best_P = P_opt
                best_support = support.copy()
                best_weights = w_opt.copy()
    
    # Try non-uniform initial weights
    for n in [3, 4, 5, 6, 8, 10]:
        if time.time() > deadline:
            break
        
        support = np.arange(n, dtype=np.float64)
        
        # Try different initial weight distributions
        for alpha in [0.5, 1.0, 2.0, 5.0]:
            if time.time() > deadline:
                break
            
            weights = np.power(np.arange(1, n+1, dtype=np.float64), alpha)
            weights = weights / weights.sum()
            
            P = _compute_P(support, weights)
            if P > best_P:
                best_P = P
                best_support = support.copy()
                best_weights = weights.copy()
            
            if time.time() > deadline:
                break
            
            time_budget = min(8.0, max(1.0, (deadline - time.time()) * 0.15))
            w_opt, P_opt = _optimize_weights(support, weights, time_budget=time_budget)
            
            if P_opt > best_P:
                best_P = P_opt
                best_support = support.copy()
                best_weights = w_opt.copy()
    
    # Try optimized support points (not just arithmetic progression)
    # Use a simple approach: try a few hand-crafted supports
    for n in [3, 4, 5, 6]:
        if time.time() > deadline:
            break
        
        # Support with a gap
        if n >= 3:
            support = np.array([0.0, 1.0, 10.0] + [0.0] * (n - 3))
            support = np.sort(support)
            weights = np.ones(n, dtype=np.float64) / n
            
            P = _compute_P(support, weights)
            if P > best_P:
                best_P = P
                best_support = support.copy()
                best_weights = weights.copy()
            
            if time.time() > deadline:
                break
            
            time_budget = min(10.0, max(1.0, (deadline - time.time()) * 0.2))
            w_opt, P_opt = _optimize_weights(support, weights, time_budget=time_budget)
            
            if P_opt > best_P:
                best_P = P_opt
                best_support = support.copy()
                best_weights = w_opt.copy()
    
    # Ensure valid output
    if best_support is None or best_weights is None:
        # Fallback to uniform on {0, 1, ..., 31}
        n = 32
        best_support = np.arange(n, dtype=np.float64)
        best_weights = np.ones(n, dtype=np.float64) / n
    else:
        # Ensure non-negative and valid
        best_support = np.abs(best_support)
        best_weights = np.abs(best_weights)
        
        # Remove zero-weight points to reduce n
        mask = best_weights > 1e-15
        if np.sum(mask) < len(best_weights):
            best_support = best_support[mask]
            best_weights = best_weights[mask]
        
        # Ensure at least 1 point
        if len(best_support) < 1:
            n = 32
            best_support = np.arange(n, dtype=np.float64)
            best_weights = np.ones(n, dtype=np.float64) / n
    
    # Final validation
    if len(best_support) != len(best_weights):
        n = 32
        best_support = np.arange(n, dtype=np.float64)
        best_weights = np.ones(n, dtype=np.float64) / n
    
    if len(best_support) < 1:
        n = 32
        best_support = np.arange(n, dtype=np.float64)
        best_weights = np.ones(n, dtype=np.float64) / n
    
    # Ensure no NaN or inf
    if not np.all(np.isfinite(best_support)) or not np.all(np.isfinite(best_weights)):
        n = 32
        best_support = np.arange(n, dtype=np.float64)
        best_weights = np.ones(n, dtype=np.float64) / n
    
    # Ensure support >= 0
    best_support = np.maximum(best_support, 0.0)
    best_weights = np.maximum(best_weights, 0.0)
    
    # Ensure sum of weights > 0
    if best_weights.sum() <= 1e-9:
        best_weights = np.ones_like(best_weights)
    
    return best_support.astype(np.float64), best_weights.astype(np.float64)

# EVOLVE-BLOCK-END
