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
    
    s_pair = s[:, None] + s[None, :]  # (n, n)
    w_pair = w[:, None] * w[None, :]  # (n, n)
    
    flat_s = s_pair.ravel()  # (n*n,)
    flat_w = w_pair.ravel()  # (n*n,)
    
    order = np.argsort(flat_s, kind='mergesort')
    sorted_s = flat_s[order]
    sorted_w = flat_w[order]
    cum_w = np.cumsum(sorted_w)
    
    targets = 2.0 * s[None, :] - s[:, None]  # (n, n), targets[k, l]
    
    flat_targets = targets.ravel()  # (n*n,)
    idx = np.searchsorted(sorted_s, flat_targets, side='left')  # (n*n,)
    
    P_X2X3 = np.zeros_like(flat_targets)
    mask = idx > 0
    P_X2X3[mask] = cum_w[idx[mask] - 1]
    
    P_X2X3 = P_X2X3.reshape(n, n)
    
    P = np.sum(w[:, None] * w[None, :] * P_X2X3)
    
    return P

def _optimize_support_and_weights(n, time_budget=5.0):
    """
    Optimize both support and weights for a given n.
    Fixes s[0]=0, s[n-1]=1.
    """
    start_time = time.time()
    
    # Parameters: s[1]..s[n-2] (n-2 vars), log_w (n vars)
    # Total dims: 2n - 2
    
    def _unpack(x):
        s_mid = x[:n-2]
        log_w = x[n-2:]
        
        # Ensure s_mid are in [0, 1] and sorted?
        # Using sigmoid to map to (0, 1)
        s_mid_val = 1.0 / (1.0 + np.exp(-s_mid * 10)) # scaled sigmoid
        # Sort s_mid
        s_mid_val = np.sort(s_mid_val)
        
        s = np.zeros(n)
        s[0] = 0.0
        s[n-1] = 1.0
        if n > 2:
            s[1:n-1] = s_mid_val
            
        # Weights
        log_w_shifted = log_w - log_w.max()
        w = np.exp(log_w_shifted)
        w = w / w.sum()
        
        return s, w

    def _neg_P(x):
        if time.time() - start_time > time_budget:
            return 1e6 # penalty for timeout
        s, w = _unpack(x)
        return -_compute_P(s, w)

    # Initial guess
    x0 = np.zeros(2*n - 2)
    # s_mid: linear spacing in sigmoid space?
    if n > 2:
        # We want s_mid to be roughly uniform in [0,1]
        # Inverse sigmoid: log(p/(1-p))
        target_s = np.linspace(0.1, 0.9, n-2)
        x0[:n-2] = np.log(target_s / (1 - target_s)) / 10.0
    # log_w: uniform
    x0[n-2:] = np.zeros(n)
    
    try:
        result = minimize(_neg_P, x0, method='L-BFGS-B', 
                         options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
        if result.fun < -0.0: # if it found something
            s_opt, w_opt = _unpack(result.x)
            P_opt = _compute_P(s_opt, w_opt)
            return s_opt, w_opt, P_opt
    except Exception:
        pass
    
    return None, None, -1.0

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
    
    # 1. Try joint optimization for small n
    for n in [3, 4, 5, 6, 7, 8]:
        if time.time() > deadline:
            break
        
        time_budget = min(30.0, max(5.0, (deadline - time.time()) * 0.2))
        s_opt, w_opt, P_opt = _optimize_support_and_weights(n, time_budget=time_budget)
        
        if s_opt is not None and P_opt > best_P:
            best_P = P_opt
            best_support = s_opt.copy()
            best_weights = w_opt.copy()
            
    # 2. Try weight optimization for fixed supports (existing logic, simplified)
    # We keep some of the original logic for robustness
    for n in [4, 5, 6, 8, 10]:
        if time.time() > deadline:
            break
        
        support = np.linspace(0, 1, n)
        weights = np.ones(n) / n
        
        P = _compute_P(support, weights)
        if P > best_P:
            best_P = P
            best_support = support.copy()
            best_weights = weights.copy()
            
        # Optimize weights only
        if time.time() > deadline:
            break
        
        def _neg_P_w(params):
            w = np.abs(params)
            s_sum = w.sum()
            if s_sum <= 1e-12: return 1e6
            return -_compute_P(support, w/s_sum)
            
        x0 = np.ones(n)
        try:
            res = minimize(_neg_P_w, x0, method='Nelder-Mead', options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10})
            w_opt = np.abs(res.x)
            s_sum = w_opt.sum()
            if s_sum > 1e-12:
                w_opt = w_opt / s_sum
                P_opt = _compute_P(support, w_opt)
                if P_opt > best_P:
                    best_P = P_opt
                    best_support = support.copy()
                    best_weights = w_opt.copy()
        except Exception:
            pass

    # Ensure valid output
    if best_support is None or best_weights is None:
        best_support = np.array([0.0, 1.0])
        best_weights = np.array([0.5, 0.5])
    else:
        best_support = np.abs(best_support)
        best_weights = np.abs(best_weights)
        best_weights = best_weights / best_weights.sum()
        
        # Remove zero-weight points
        mask = best_weights > 1e-15
        if np.sum(mask) < len(best_weights):
            best_support = best_support[mask]
            best_weights = best_weights[mask]
            
        if len(best_support) < 1:
            best_support = np.array([0.0, 1.0])
            best_weights = np.array([0.5, 0.5])

    if not np.all(np.isfinite(best_support)) or not np.all(np.isfinite(best_weights)):
        best_support = np.array([0.0, 1.0])
        best_weights = np.array([0.5, 0.5])
        
    best_support = np.maximum(best_support, 0.0)
    best_weights = np.maximum(best_weights, 0.0)
    
    return best_support.astype(np.float64), best_weights.astype(np.float64)

# EVOLVE-BLOCK-END
