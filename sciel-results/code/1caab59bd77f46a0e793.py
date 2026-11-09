# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_P_fast(support, weights):
    """
    Compute P[X1+X2+X3 < 2*X4] for discrete measure with support and weights.
    Uses vectorized pairwise sums for speed.
    """
    s = np.asarray(support, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    n = len(s)
    
    # Build pairwise sum matrix: sum[i,j] = s[i] + s[j]
    s_col = s[:, np.newaxis]
    s_row = s[np.newaxis, :]
    pair_sums = s_col + s_row  # shape (n, n)
    pair_probs = np.outer(w, w)  # shape (n, n)
    
    P = 0.0
    for k in range(n):
        for l in range(n):
            threshold = 2.0 * s[l] - s[k]
            # Sum of pair_probs where pair_sums < threshold
            mask = pair_sums < threshold
            triple_prob = np.sum(pair_probs * mask)
            P += w[k] * w[l] * triple_prob
    return P

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.
    
    Strategy: Use a geometric-like distribution concentrated near 0 with a long tail,
    or optimize the weights on a fixed support.
    
    Returns:
        support: np.ndarray of shape (n,), non-negative support points of the measure.
        weights: np.ndarray of shape (n,), non-negative weights.
    """
    start_time = time.time()
    deadline = 600.0
    
    # Start with the baseline
    n_base = 32
    support_base = np.arange(n_base, dtype=np.float64)
    weights_base = np.ones(n_base, dtype=np.float64) / n_base
    best_P = _compute_P_fast(support_base, weights_base)
    best_support = support_base.copy()
    best_weights = weights_base.copy()
    
    # Try different support structures
    # 1. Geometric-like support with optimized weights
    # 2. Support concentrated near 0
    
    # Try a support that's more concentrated near 0
    # Use exponential spacing
    for n_try in [16, 32, 64]:
        if time.time() > deadline - 60:
            break
        
        # Exponential spacing: s_i = c * (r^i - 1) or similar
        # Let's try s_i = i^2 / n or s_i = exp(i/n * log(M))
        
        # Option A: Quadratic spacing
        support_quad = np.arange(n_try, dtype=np.float64) ** 2 / n_try
        weights_quad = np.ones(n_try, dtype=np.float64) / n_try
        P_quad = _compute_P_fast(support_quad, weights_quad)
        if P_quad > best_P:
            best_P = P_quad
            best_support = support_quad.copy()
            best_weights = weights_quad.copy()
        
        # Option B: Exponential spacing
        if n_try > 1:
            r = 2.0
            support_exp = (r ** np.arange(n_try) - 1.0) / (r ** (n_try - 1) - 1.0)
            weights_exp = np.ones(n_try, dtype=np.float64) / n_try
            P_exp = _compute_P_fast(support_exp, weights_exp)
            if P_exp > best_P:
                best_P = P_exp
                best_support = support_exp.copy()
                best_weights = weights_exp.copy()
        
        # Option C: Log-spaced
        support_log = np.logspace(0, 2, n_try)
        weights_log = np.ones(n_try, dtype=np.float64) / n_try
        P_log = _compute_P_fast(support_log, weights_log)
        if P_log > best_P:
            best_P = P_log
            best_support = support_log.copy()
            best_weights = weights_log.copy()
    
    # Optimize weights on the best support found
    if time.time() < deadline - 120:
        from scipy.optimize import minimize
        
        s_opt = best_support.copy()
        n_opt = len(s_opt)
        
        def neg_P(params):
            w = np.exp(params)  # Softplus-like to ensure positivity
            w = w / w.sum()
            return -_compute_P_fast(s_opt, w)
        
        # Initial guess: uniform
        w0 = np.ones(n_opt) / n_opt
        x0 = np.log(w0)
        
        result = minimize(neg_P, x0, method='Nelder-Mead', 
                          options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
        
        if result.fun < -best_P:
            w_opt = np.exp(result.x)
            w_opt = w_opt / w_opt.sum()
            best_P = -result.fun
            best_weights = w_opt
            best_support = s_opt.copy()
    
    # Try a few more support structures with optimized weights
    if time.time() < deadline - 60:
        from scipy.optimize import minimize
        
        # Try support with a few points
        for n_pts in [8, 12]:
            # Try support: [0, 1, 2, ..., n_pts-1] with geometric weights
            s_test = np.arange(n_pts, dtype=np.float64)
            
            def neg_P_test(params):
                w = np.exp(params)
                w = w / w.sum()
                return -_compute_P_fast(s_test, w)
            
            w0_test = np.ones(n_pts) / n_pts
            x0_test = np.log(w0_test)
            
            result = minimize(neg_P_test, x0_test, method='Nelder-Mead',
                             options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
            
            if result.fun < -best_P:
                w_test = np.exp(result.x)
                w_test = w_test / w_test.sum()
                best_P = -result.fun
                best_weights = w_test
                best_support = s_test.copy()
    
    # Ensure validity
    best_support = np.asarray(best_support, dtype=np.float64)
    best_weights = np.asarray(best_weights, dtype=np.float64)
    
    # Clean up any negative values
    best_weights = np.maximum(best_weights, 0.0)
    if best_weights.sum() <= 1e-9:
        # Fallback to uniform
        n = len(best_support)
        best_weights = np.ones(n, dtype=np.float64) / n
    
    # Ensure support is non-negative
    best_support = np.maximum(best_support, 0.0)
    
    return best_support, best_weights

# EVOLVE-BLOCK-END
