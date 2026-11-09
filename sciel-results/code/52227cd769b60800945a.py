# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_P_fast(support, weights):
    """
    Compute P[X1 + X2 + X3 < 2*X4] efficiently using O(n^2 log n) searchsorted.
    """
    n = len(support)
    if n == 0:
        return 0.0
    
    w_sum = np.sum(weights)
    if w_sum <= 1e-9:
        return 0.0
    w = weights / w_sum
    s = support.astype(np.float64)
    
    if n == 1:
        return 0.0
    
    s_col = s.reshape(-1, 1)
    s_row = s.reshape(1, -1)
    pair_sums = s_col + s_row
    pair_probs = np.outer(w, w)
    
    sums_flat = pair_sums.flatten()
    probs_flat = pair_probs.flatten()
    
    sort_idx = np.argsort(sums_flat)
    sorted_sums = sums_flat[sort_idx]
    sorted_probs = probs_flat[sort_idx]
    cum_probs = np.cumsum(sorted_probs)
    
    if cum_probs[-1] < 1.0:
        if np.isclose(cum_probs[-1], 1.0, atol=1e-9):
            cum_probs[-1] = 1.0
    
    thresholds = 2.0 * s_row - s_col
    indices = np.searchsorted(sorted_sums, thresholds, side='left')
    cum_probs_with_zero = np.concatenate(([0.0], cum_probs))
    cdf_values = cum_probs_with_zero[indices]
    
    w_k = w.reshape(-1, 1)
    w_l = w.reshape(1, -1)
    
    P = np.sum(w_k * w_l * cdf_values)
    return float(P)

def _objective_weights(x, s_fixed):
    """
    Objective for optimizing weights given a fixed geometric support.
    x = log-weights.
    """
    n = len(s_fixed)
    log_w = x
    max_log_w = np.max(log_w)
    exp_w = np.exp(log_w - max_log_w)
    w_sum = np.sum(exp_w)
    if w_sum <= 1e-300:
        return 1e6
    w = exp_w / w_sum
    
    if np.any(~np.isfinite(w)):
        return 1e6
        
    try:
        P = _compute_P_fast(s_fixed, w)
    except Exception:
        return 1e6
        
    if not np.isfinite(P):
        return 1e6
        
    return -P

def _optimize_weights_for_support(s_fixed, t_start, deadline):
    """
    Optimize weights for a fixed support using Powell method.
    Returns (best_weights, best_P).
    """
    n = len(s_fixed)
    if n == 0:
        return np.array([]), 0.0
    
    # Initialize with a few different weight distributions
    best_w = None
    best_P = -np.inf
    
    # Strategy 1: Uniform weights
    x0_1 = np.zeros(n)
    
    # Strategy 2: Linear decay (heavier on smaller values)
    x0_2 = np.linspace(0, 2, n)
    
    # Strategy 3: Linear growth (heavier on larger values)
    x0_3 = np.linspace(-2, 0, n)
    
    # Strategy 4: Convex decay
    x0_4 = np.linspace(0, 1, n)**2 * 2
    
    # Strategy 5: Random
    np.random.seed(123)
    x0_5 = np.random.uniform(-1, 1, n)
    
    initializers = [x0_1, x0_2, x0_3, x0_4, x0_5]
    
    for x0 in initializers:
        if time.time() > deadline - 5:
            break
            
        try:
            res = minimize(
                _objective_weights, 
                x0, 
                args=(s_fixed,), 
                method='Powell', 
                options={'maxiter': 200, 'xtol': 1e-8, 'ftol': 1e-8}
            )
            
            if res.fun < 1e5:
                log_w_opt = res.x
                max_log_w = np.max(log_w_opt)
                exp_w = np.exp(log_w_opt - max_log_w)
                w_opt = exp_w / np.sum(exp_w)
                
                # Ensure validity
                if np.all(w_opt >= 0) and np.all(np.isfinite(w_opt)):
                    P_val = _compute_P_fast(s_fixed, w_opt)
                    if P_val > best_P:
                        best_P = P_val
                        best_w = w_opt.copy()
        except Exception:
            pass
            
    if best_w is None:
        # Fallback to uniform
        best_w = np.ones(n) / n
        best_P = _compute_P_fast(s_fixed, best_w)
        
    return best_w, best_P

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game.
    Strategy: Structured search over geometric progression supports (s_i = a * r^i)
    with optimized weights via Powell method.
    """
    t_start = time.time()
    deadline = t_start + 600.0 - 10.0  # Safety margin
    
    best_s = None
    best_w = None
    best_P = -np.inf
    
    # Parameters for geometric progression: s_i = a * r^i
    # n: number of points
    # a: scale (s_0)
    # r: ratio
    
    # Search space exploration
    # We want to explore a range of r and a.
    # r > 1 implies increasing support.
    # r < 1 implies decreasing support.
    
    # Let's try a grid of r and a, then refine.
    
    # Initial grid for r
    # r from 0.5 to 2.0
    # Log-spaced for better coverage
    r_values = np.geomspace(0.5, 2.0, 40)
    
    # a values
    # a from 0.01 to 5.0
    a_values = np.geomspace(0.01, 5.0, 40)
    
    # n values to try
    n_values = [3, 4, 5, 6, 8, 10]
    
    # We will iterate through n, then r, then a.
    # To save time, we'll do a coarse grid first, then refine the best (r, a) pair.
    
    # Phase 1: Coarse Grid Search
    # For each n, find the best (r, a) combination based on a quick weight optimization
    # or even just uniform weights to guide the search.
    
    # However, weight optimization is expensive.
    # Let's use a proxy: Uniform weights P value.
    # Then, for the top candidates, do full weight optimization.
    
    candidate_list = []
    
    for n in n_values:
        if time.time() > deadline - 30:
            break
            
        for r in r_values:
            if time.time() > deadline - 30:
                break
                
            for a in a_values:
                # Generate support
                s = a * (r ** np.arange(n))
                
                # Check validity: s_i < -1e-9 is invalid. s_i >= 0 is required.
                # Since a > 0 and r > 0, s > 0.
                # Check for overflow/inf
                if not np.all(np.isfinite(s)):
                    continue
                
                # Quick check: Uniform weights
                w_uniform = np.ones(n) / n
                P_quick = _compute_P_fast(s, w_uniform)
                
                candidate_list.append((P_quick, n, r, a))
                
    # Sort candidates by P_quick
    candidate_list.sort(key=lambda x: x[0], reverse=True)
    
    # Phase 2: Refine Top Candidates with Weight Optimization
    # Take top 50 candidates
    top_candidates = candidate_list[:50]
    
    for P_quick, n, r, a in top_candidates:
        if time.time() > deadline - 20:
            break
            
        s = a * (r ** np.arange(n))
        
        # Refine r and a slightly around this point
        # Let's do a small local grid search around (r, a) with weight optimization
        # Actually, let's just run weight optimization on the exact point first.
        
        w_opt, P_opt = _optimize_weights_for_support(s, t_start, deadline)
        
        if P_opt > best_P:
            best_P = P_opt
            best_s = s.copy()
            best_w = w_opt.copy()
            
    # Phase 3: Local Refinement of the Best Candidate
    # If we found a good candidate, refine r and a using a finer grid or optimization.
    
    if best_s is not None and time.time() < deadline - 10:
        n_best = len(best_s)
        
        # Extract current r and a from best_s
        # s_i = a * r^i => s_1/s_0 = r, s_0 = a
        if n_best > 1 and best_s[0] > 0:
            r_curr = best_s[1] / best_s[0]
            a_curr = best_s[0]
            
            # Refine r and a
            # Small perturbations
            r_range = np.linspace(r_curr * 0.9, r_curr * 1.1, 11)
            a_range = np.linspace(a_curr * 0.9, a_curr * 1.1, 11)
            
            for r_try in r_range:
                if time.time() > deadline - 10:
                    break
                for a_try in a_range:
                    if time.time() > deadline - 10:
                        break
                        
                    s_try = a_try * (r_try ** np.arange(n_best))
                    if not np.all(np.isfinite(s_try)):
                        continue
                        
                    w_try, P_try = _optimize_weights_for_support(s_try, t_start, deadline)
                    if P_try > best_P:
                        best_P = P_try
                        best_s = s_try.copy()
                        best_w = w_try.copy()
            
            # Also try varying n slightly
            # n_best - 1, n_best + 1
            for n_try in [n_best - 1, n_best + 1, n_best + 2]:
                if n_try < 2 or n_try > 12:
                    continue
                if time.time() > deadline - 10:
                    break
                    
                # Use the same r and a (scaled appropriately?)
                # Just use the current r and a
                s_try = a_curr * (r_curr ** np.arange(n_try))
                if not np.all(np.isfinite(s_try)):
                    continue
                    
                w_try, P_try = _optimize_weights_for_support(s_try, t_start, deadline)
                if P_try > best_P:
                    best_P = P_try
                    best_s = s_try.copy()
                    best_w = w_try.copy()

    # Fallback
    if best_s is None:
        best_s = np.array([0.0, 1.0])
        best_w = np.array([0.5, 0.5])
        
    # Validation and formatting
    best_s = np.asarray(best_s, dtype=np.float64).flatten()
    best_w = np.asarray(best_w, dtype=np.float64).flatten()
    
    if len(best_s) == 0 or len(best_w) == 0:
        return np.array([0.0, 1.0]), np.array([0.5, 0.5])
        
    if len(best_s) != len(best_w):
        return np.array([0.0, 1.0]), np.array([0.5, 0.5])
        
    best_s = np.maximum(best_s, 0.0)
    best_w = np.maximum(best_w, 0.0)
    
    w_sum = np.sum(best_w)
    if w_sum <= 1e-9:
        return np.array([0.0, 1.0]), np.array([0.5, 0.5])
        
    best_w = best_w / w_sum
    
    if not (np.all(np.isfinite(best_s)) and np.all(np.isfinite(best_w))):
        return np.array([0.0, 1.0]), np.array([0.5, 0.5])
        
    sort_idx = np.argsort(best_s)
    best_s = best_s[sort_idx]
    best_w = best_w[sort_idx]
    
    return best_s, best_w

# EVOLVE-BLOCK-END
