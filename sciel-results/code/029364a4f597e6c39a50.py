# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_P_fast(support, weights):
    """
    Compute P[X1 + X2 + X3 < 2*X4] efficiently.
    P = sum_{k,l} w_k w_l * Prob[X1+X2 < 2*s_l - s_k]
    """
    n = len(support)
    if n == 0:
        return 0.0
    
    w_sum = np.sum(weights)
    if w_sum <= 1e-9:
        return 0.0
    w = weights / w_sum
    s = support.astype(np.float64)
    
    # Generate all pair sums and their probabilities
    s_col = s.reshape(-1, 1)
    s_row = s.reshape(1, -1)
    pair_sums = s_col + s_row  # (n, n)
    pair_probs = np.outer(w, w) # (n, n)
    
    # Flatten
    sums_flat = pair_sums.flatten() # (n^2,)
    probs_flat = pair_probs.flatten() # (n^2,)
    
    # Sort sums and compute cumulative probabilities
    sort_idx = np.argsort(sums_flat)
    sorted_sums = sums_flat[sort_idx]
    sorted_probs = probs_flat[sort_idx]
    cum_probs = np.cumsum(sorted_probs)
    
    # Ensure cum_probs ends at 1.0 (floating point safety)
    if cum_probs[-1] < 1.0:
        if np.isclose(cum_probs[-1], 1.0, atol=1e-9):
             cum_probs[-1] = 1.0
    
    # Compute P = sum_{k,l} w_k w_l * CDF(2*s_l - s_k)
    thresholds = 2.0 * s_row - s_col  # (n, n)
    
    # For each threshold, find the count of pair_sums < threshold
    # searchsorted with side='left' gives the index of the first element >= threshold.
    # We want sum of probs for elements < threshold.
    indices = np.searchsorted(sorted_sums, thresholds, side='left')
    
    # Get CDF values
    cum_probs_with_zero = np.concatenate(([0.0], cum_probs))
    cdf_values = cum_probs_with_zero[indices]
    
    # Sum over k, l with weights w_k w_l
    w_k = w.reshape(-1, 1)
    w_l = w.reshape(1, -1)
    
    P = np.sum(w_k * w_l * cdf_values)
    
    return float(P)

def _objective(log_w, log_s):
    """
    Objective function for minimization (minimizes -P).
    log_w: raw log-weights (softmaxed internally)
    log_s: raw log-support (exponentiated internally, clipped to 0)
    """
    # Decode weights
    log_w_max = np.max(log_w)
    w = np.exp(log_w - log_w_max)
    w_sum = np.sum(w)
    if w_sum < 1e-9:
        return 0.0
    w = w / w_sum
    
    # Decode support
    s = np.exp(log_s)
    # Handle underflow to 0 naturally, but ensure no negatives
    s = np.maximum(s, 0.0)
    
    # Sort to maintain consistent ordering for numerical stability in P calculation
    # Note: P is invariant to permutation of (s, w) pairs, but sorting helps with
    # potential issues in searchsorted if the distribution of sums is not monotonic
    # relative to the index, although mathematically it shouldn't matter.
    # However, sorting s ensures that if we use s for anything else, it's consistent.
    # Actually, _compute_P_fast handles any order. Sorting is optional but safe.
    sort_idx = np.argsort(s)
    s_sorted = s[sort_idx]
    w_sorted = w[sort_idx]
    
    P = _compute_P_fast(s_sorted, w_sorted)
    return -P

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game.
    Strategy: Multi-start direct optimization on (log_w, log_s) using L-BFGS-B,
    followed by a high-precision polish pass.
    """
    t_start = time.time()
    deadline = t_start + 600.0
    
    best_P_global = -1.0
    best_support_global = None
    best_weights_global = None
    
    # Fallback: simple baseline (2-point optimal-ish)
    # For 2 points [0, 1], optimal alpha is ~0.382 (root of 1+2a-4a^2+... wait)
    # f(a) = a(1-a)^2(1+2a). f'(a) = 0.
    # Max is around a=0.38.
    s_base = np.array([0.0, 1.0])
    w_base = np.array([0.618, 0.382])
    P_base = _compute_P_fast(s_base, w_base)
    best_P_global = P_base
    best_support_global = s_base.copy()
    best_weights_global = w_base.copy()
    
    def update_best(s, w, P_val):
        nonlocal best_P_global, best_support_global, best_weights_global
        if P_val > best_P_global:
            best_P_global = P_val
            best_support_global = s.copy()
            best_weights_global = w.copy()
            
    # Define the optimization problem for a fixed n
    # We will try n=2, n=3, n=4.
    # Parameters: log_w (n), log_s (n).
    # Constraints:
    # - Weights are normalized via softmax, so no explicit constraint needed on sum.
    # - Support s_i >= 0. Using log_s = log(s_i + eps) or just log_s with s=exp(log_s)
    #   ensures s > 0. If we want to allow s=0, we can use softplus or just let
    #   exp(log_s) go to 0. However, if s=0 is optimal, log_s -> -inf.
    #   L-BFGS-B works on bounded or unbounded domains. 
    #   To allow s=0, we can parameterize s_i = exp(log_s_i). 
    #   If the optimizer wants s_i=0, it will drive log_s_i to -inf.
    #   This can cause numerical issues. 
    #   Alternative: s_i = softplus(log_s_i) = log(1+exp(log_s_i)).
    #   softplus(0) = log(2) ~ 0.69. softplus(-10) ~ 0.
    #   This maps R to (0, inf). It never reaches 0 exactly, but gets arbitrarily close.
    #   Given the problem allows s >= 0, and the optimal 2-point solution has s_1=0,
    #   we should be able to get close enough.
    #   However, if the true optimum is exactly 0, we might miss it slightly.
    #   Let's stick to exp(log_s) but clip the final result to 0.
    #   During optimization, if log_s becomes very negative, exp underflows to 0.
    
    def run_optimization(n, initial_log_w, initial_log_s, time_limit):
        t_opt_start = time.time()
        
        # Initial evaluation
        current_P = -_objective(initial_log_w, initial_log_s)
        
        # Options for L-BFGS-B
        options = {
            'maxiter': 1000,
            'ftol': 1e-10,
            'gtol': 1e-8,
            'disp': False
        }
        
        # We don't have bounds on log_w or log_s in the standard sense,
        # but we can set broad bounds to prevent overflow/underflow issues.
        # log_w: softmax is scale invariant, so we can fix one or just let it float.
        # To prevent drift, we can normalize log_w periodically? 
        # L-BFGS-B handles this usually.
        # log_s: exp(log_s) can overflow if log_s > 700.
        # Let's set bounds: log_s in [-50, 50]. exp(-50) ~ 1e-22 (effectively 0).
        # exp(50) ~ 5e21.
        
        bounds = [(-50.0, 50.0)] * (2 * n)
        
        # Flatten parameters
        x0 = np.concatenate([initial_log_w, initial_log_s])
        
        def obj_func(x):
            if time.time() > t_opt_start + time_limit:
                return 1e10 # Penalty for timeout
            lw = x[:n]
            ls = x[n:]
            return _objective(lw, ls)
        
        try:
            res = minimize(obj_func, x0, method='L-BFGS-B', bounds=bounds, options=options)
            if res.success or res.fun < obj_func(x0):
                final_log_w = res.x[:n]
                final_log_s = res.x[n:]
                
                # Decode
                log_w_max = np.max(final_log_w)
                w = np.exp(final_log_w - log_w_max)
                w = w / np.sum(w)
                
                s = np.exp(final_log_s)
                s = np.maximum(s, 0.0)
                
                # Sort
                sort_idx = np.argsort(s)
                s_sorted = s[sort_idx]
                w_sorted = w[sort_idx]
                
                P_val = _compute_P_fast(s_sorted, w_sorted)
                return s_sorted, w_sorted, P_val
        except Exception as e:
            pass
            
        return None, None, -1.0

    # --- Multi-Start Search ---
    
    # We will try n=2, 3, 4.
    # For each n, we will try several random initializations.
    
    np.random.seed(42)
    
    for n in [2, 3, 4]:
        if time.time() > deadline - 120:
            break
            
        # Number of restarts
        num_restarts = 5 if n <= 3 else 3
        
        for r in range(num_restarts):
            if time.time() > deadline - 60:
                break
                
            # Initialize log_w
            # Start with roughly uniform weights
            init_w = np.ones(n) / n
            # Add some noise
            init_w = init_w + np.random.uniform(-0.1, 0.1, n)
            init_w = np.maximum(init_w, 1e-3)
            init_w = init_w / np.sum(init_w)
            init_log_w = np.log(init_w)
            
            # Initialize log_s
            # Spread support points
            if n == 2:
                init_s = np.array([0.0, 1.0])
            elif n == 3:
                init_s = np.array([0.0, 0.5, 1.0])
            else:
                init_s = np.linspace(0.0, 1.0, n)
            
            # Add noise to support
            init_s = init_s + np.random.uniform(-0.1, 0.1, n)
            init_s = np.maximum(init_s, 1e-6) # Ensure positive for log
            init_log_s = np.log(init_s)
            
            # Run optimization
            s_opt, w_opt, P_opt = run_optimization(n, init_log_w, init_log_s, time_limit=60.0)
            
            if s_opt is not None and P_opt > best_P_global:
                update_best(s_opt, w_opt, P_opt)
                
    # --- High-Precision Polish Pass ---
    # Take the best solution found and run a very tight optimization.
    if best_support_global is not None and time.time() < deadline - 30:
        n_best = len(best_support_global)
        
        # Initialize from best
        s_safe = np.maximum(best_support_global, 1e-10)
        w_safe = np.maximum(best_weights_global, 1e-10)
        log_s0 = np.log(s_safe)
        log_w0 = np.log(w_safe)
        
        # Tighter options
        # We use a dedicated function for the polish to ensure we don't hit the generic timeout
        # and to allow specific tighter tolerances.
        
        t_polish_start = time.time()
        polish_time_limit = 60.0
        
        bounds = [(-50.0, 50.0)] * (2 * n_best)
        x0 = np.concatenate([log_w0, log_s0])
        
        def obj_polish(x):
            if time.time() > t_polish_start + polish_time_limit:
                return 1e10
            lw = x[:n_best]
            ls = x[n_best:]
            return _objective(lw, ls)
        
        options_polish = {
            'maxiter': 5000,
            'ftol': 1e-12,
            'gtol': 1e-10,
            'disp': False
        }
        
        try:
            res_polish = minimize(obj_polish, x0, method='L-BFGS-B', bounds=bounds, options=options_polish)
            if res_polish.success:
                final_log_w = res_polish.x[:n_best]
                final_log_s = res_polish.x[n_best:]
                
                log_w_max = np.max(final_log_w)
                w = np.exp(final_log_w - log_w_max)
                w = w / np.sum(w)
                
                s = np.exp(final_log_s)
                s = np.maximum(s, 0.0)
                
                sort_idx = np.argsort(s)
                s_sorted = s[sort_idx]
                w_sorted = w[sort_idx]
                
                P_val = _compute_P_fast(s_sorted, w_sorted)
                update_best(s_sorted, w_sorted, P_val)
        except Exception:
            pass

    # Final Validation
    if best_support_global is None or len(best_support_global) == 0:
        return s_base, w_base
        
    best_support_global = np.maximum(best_support_global, 0.0)
    best_weights_global = np.maximum(best_weights_global, 0.0)
    
    w_sum = np.sum(best_weights_global)
    if w_sum <= 1e-9:
        return s_base, w_base
        
    best_weights_global = best_weights_global / w_sum
    
    if not (np.all(np.isfinite(best_support_global)) and np.all(np.isfinite(best_weights_global))):
        return s_base, w_base
        
    return best_support_global, best_weights_global

# EVOLVE-BLOCK-END
