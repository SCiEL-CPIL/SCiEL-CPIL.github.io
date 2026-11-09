# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_P_fast(support, weights):
    """
    Compute P[X1 + X2 + X3 < 2*X4] efficiently.
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

def _objective(x, n):
    """
    Objective function for scipy.optimize.minimize.
    x = [s_0, ..., s_{n-1}, log(w_0), ..., log(w_{n-1})]
    We minimize -P.
    Constraints: s_i >= 0, w_i > 0.
    """
    s = x[:n]
    log_w = x[n:]
    
    # Enforce s >= 0 by clipping (or using bounds)
    # L-BFGS-B handles bounds, but let's be safe for internal computations
    s_eff = np.maximum(s, 0.0)
    
    # Softmax for weights to ensure positive and sum to 1
    # Using log-sum-exp for stability
    max_log_w = np.max(log_w)
    exp_w = np.exp(log_w - max_log_w)
    w = exp_w / np.sum(exp_w)
    
    # Check for degenerate cases
    if np.any(~np.isfinite(s_eff)) or np.any(~np.isfinite(w)):
        return 1e6
    
    # If support points are too close, P is unstable/undefined in limit
    # But L-BFGS-B should navigate around this if initialized well
    try:
        P = _compute_P_fast(s_eff, w)
    except Exception:
        return 1e6
        
    if not np.isfinite(P):
        return 1e6
        
    return -P

def _run_optimization(n, x0, t_start, time_limit):
    """Run L-BFGS-B optimization for a fixed n."""
    deadline = t_start + time_limit
    
    s0 = x0[:n]
    log_w0 = x0[n:]
    
    # Bounds: s in [0, 10], log_w in [-10, 10]
    # The scale of s matters. Let's assume s is in [0, 5] roughly.
    # If the optimal s is very large or small, we might miss it.
    # But for this problem, scaling s scales the inequality linearly, 
    # so the optimal shape is scale-invariant in terms of P, but P depends on relative positions.
    # Actually, P[X1+X2+X3 < 2*X4] is scale invariant if we scale all s.
    # So we can fix one scale, e.g., max(s) = 1, or just let it optimize in [0, 10].
    
    bounds = [(0.0, 10.0)] * n + [(-10.0, 10.0)] * n
    
    result = None
    try:
        result = minimize(
            _objective, 
            x0, 
            args=(n,), 
            method='L-BFGS-B', 
            bounds=bounds, 
            options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-6, 'maxls': 50}
        )
    except Exception as e:
        pass
        
    if result is not None and result.success:
        x_opt = result.x
        s_opt = np.maximum(x_opt[:n], 0.0)
        log_w_opt = x_opt[n:]
        exp_w = np.exp(log_w_opt - np.max(log_w_opt))
        w_opt = exp_w / np.sum(exp_w)
        
        # Clean up tiny weights
        w_opt[w_opt < 1e-6] = 0.0
        if np.sum(w_opt) < 1e-9:
            return None, None, 0.0
        w_opt = w_opt / np.sum(w_opt)
        
        # Remove atoms with negligible weight to simplify
        mask = w_opt > 1e-5
        s_final = s_opt[mask]
        w_final = w_opt[mask]
        
        if len(s_final) == 0:
            return None, None, 0.0
            
        P_final = _compute_P_fast(s_final, w_final)
        return s_final, w_final, P_final
    else:
        return None, None, 0.0

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game.
    Strategy: Multi-start L-BFGS-B optimization with various initializations.
    """
    t_start = time.time()
    deadline = t_start + 590.0  # Safety margin
    
    best_s = None
    best_w = None
    best_P = -1.0
    
    # Define a set of initializations for different n
    # n is the number of atoms. We try small n first, then increase.
    # The problem is discrete, so small n might be optimal, but continuous approximation
    # suggests finer grids might be better. However, the metric is exact for discrete.
    
    np.random.seed(42)
    
    # Strategy: Try n=2, 3, 4, 5, 6, 8, 10 with multiple random starts each.
    # We allocate time budget based on n. Smaller n is faster to evaluate.
    
    n_candidates = [2, 3, 4, 5, 6, 8, 10, 12]
    
    # Time allocation: give more time to smaller n initially, then refine.
    # Or just iterate through them and stop when time runs out.
    
    # Let's do a few passes.
    # Pass 1: Coarse search with small n
    # Pass 2: Refine best small n, try larger n
    
    # We will generate initial points on the fly.
    
    for n in n_candidates:
        if time.time() > deadline - 30:
            break
            
        # Number of random starts for this n
        # Smaller n can afford more starts
        n_starts = max(2, 20 - n)
        
        for start_idx in range(n_starts):
            if time.time() > deadline - 10:
                break
                
            # Generate initial x
            # Support: random points in [0, 5]
            # Weights: random log-weights
            
            # Different initialization strategies
            if start_idx == 0:
                # Uniform grid
                s0 = np.linspace(0.1, 2.0, n)
            elif start_idx == 1:
                # Geometric progression
                s0 = np.geomspace(0.1, 2.0, n)
            elif start_idx == 2:
                # Clustered
                s0 = np.sort(np.random.uniform(0, 1, n))
            else:
                # Random
                s0 = np.sort(np.random.uniform(0, 3, n))
                
            # Log weights: random
            log_w0 = np.random.uniform(-2, 2, n)
            
            x0 = np.concatenate([s0, log_w0])
            
            # Time limit for this single optimization
            # Smaller n is faster. Let's give 10s for n<=5, 5s for n>5
            time_limit = 10.0 if n <= 5 else 5.0
            
            s_opt, w_opt, P_opt = _run_optimization(n, x0, time.time(), time_limit)
            
            if s_opt is not None and P_opt > best_P:
                best_P = P_opt
                best_s = s_opt
                best_w = w_opt
                
                # If we found a good solution, we can try to refine it with more iterations
                # or use it as a seed for larger n (by adding small weights to new points)
                
    # If no valid solution found, return a baseline
    if best_s is None:
        # Baseline: 2 points
        s_base = np.array([0.0, 1.0])
        w_base = np.array([0.5, 0.5])
        return s_base, w_base
        
    # Final polish: Try to optimize the best found solution again with more iterations
    # or with a different optimizer (Nelder-Mead) if L-BFGS-B got stuck
    if time.time() < deadline - 20:
        n_best = len(best_s)
        s_curr = best_s.copy()
        w_curr = best_w.copy()
        
        # Convert to log-weights for the optimizer
        # Ensure w_curr > 0
        w_curr = np.maximum(w_curr, 1e-10)
        w_curr = w_curr / np.sum(w_curr)
        log_w_curr = np.log(w_curr)
        
        x0 = np.concatenate([s_curr, log_w_curr])
        
        # Try Nelder-Mead for a local refinement
        try:
            # Nelder-Mead doesn't support bounds well, so we clip in objective
            # But our objective clips s and uses softmax for w, so it's safe.
            res = minimize(
                _objective, 
                x0, 
                args=(n_best,), 
                method='Nelder-Mead', 
                options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6}
            )
            if res.success:
                x_opt = res.x
                s_opt = np.maximum(x_opt[:n_best], 0.0)
                log_w_opt = x_opt[n_best:]
                exp_w = np.exp(log_w_opt - np.max(log_w_opt))
                w_opt = exp_w / np.sum(exp_w)
                
                w_opt[w_opt < 1e-6] = 0.0
                if np.sum(w_opt) > 1e-9:
                    w_opt = w_opt / np.sum(w_opt)
                    mask = w_opt > 1e-5
                    s_final = s_opt[mask]
                    w_final = w_opt[mask]
                    
                    if len(s_final) > 0:
                        P_final = _compute_P_fast(s_final, w_final)
                        if P_final > best_P:
                            best_P = P_final
                            best_s = s_final
                            best_w = w_final
        except Exception:
            pass
            
    # Validation and formatting
    best_s = np.asarray(best_s, dtype=np.float64).flatten()
    best_w = np.asarray(best_w, dtype=np.float64).flatten()
    
    if len(best_s) == 0 or len(best_w) == 0:
        return np.array([0.0, 1.0]), np.array([0.5, 0.5])
        
    if len(best_s) != len(best_w):
        return np.array([0.0, 1.0]), np.array([0.5, 0.5])
        
    # Ensure non-negative
    best_s = np.maximum(best_s, 0.0)
    best_w = np.maximum(best_w, 0.0)
    
    w_sum = np.sum(best_w)
    if w_sum <= 1e-9:
        return np.array([0.0, 1.0]), np.array([0.5, 0.5])
        
    best_w = best_w / w_sum
    
    if not (np.all(np.isfinite(best_s)) and np.all(np.isfinite(best_w))):
        return np.array([0.0, 1.0]), np.array([0.5, 0.5])
        
    # Sort by support
    sort_idx = np.argsort(best_s)
    best_s = best_s[sort_idx]
    best_w = best_w[sort_idx]
    
    return best_s, best_w

# EVOLVE-BLOCK-END
