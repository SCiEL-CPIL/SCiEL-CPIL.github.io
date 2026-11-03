"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 100

# EVOLVE-BLOCK-START
def _max_overlap(h):
    """Compute max over k of sum_i h[i] * (1 - h[(i+k) % N])."""
    h = np.asarray(h)
    N = len(h)
    max_ov = 0.0
    for k in range(N):
        shifted = np.roll(h, k)
        ov = np.sum(h * (1.0 - shifted))
        if ov > max_ov:
            max_ov = ov
    return max_ov

def _sum_constraint_violation(h, target=None):
    if target is None:
        target = N / 2.0
    return np.sum(h) - target

def _fix_sum(h, target=None, tol=1e-9):
    """Adjust h so that sum(h) == target, keeping h in [0,1]."""
    if target is None:
        target = N / 2.0
    h = np.asarray(h, dtype=float).copy()
    diff = target - np.sum(h)
    if abs(diff) <= tol:
        # Clip just in case
        return np.clip(h, 0.0, 1.0)
    # Try to add diff by adjusting elements with the most "room"
    # We'll do a simple water-filling: add to elements farthest from 0.5 in the direction of the sign of diff
    if diff > 0:
        # Need to increase sum: add to elements with smallest h (most room to increase)
        idxs = np.argsort(h)  # ascending h
    else:
        # Need to decrease sum: remove from elements with largest h (most room to decrease)
        idxs = np.argsort(h)[::-1]  # descending h
    
    remaining = diff
    for idx in idxs:
        if remaining == 0.0:
            break
        if diff > 0:
            can_add = 1.0 - h[idx]
            add = min(can_add, remaining)
            h[idx] += add
            remaining -= add
        else:
            can_remove = h[idx]
            remove = min(can_remove, -remaining)
            h[idx] -= remove
            remaining += remove
    return np.clip(h, 0.0, 1.0)

def _smooth_max_overlap(h, beta=50.0):
    """Smooth surrogate for max overlap using log-sum-exp."""
    h = np.asarray(h, dtype=float)
    N = len(h)
    ov_values = np.empty(N)
    for k in range(N):
        shifted = np.roll(h, k)
        ov_values[k] = np.sum(h * (1.0 - shifted))
    # log-sum-exp for stability
    m = np.max(ov_values)
    if m == 0:
        return 0.0
    log_sum_exp = m + np.log(np.sum(np.exp(ov_values - m) / beta))
    return log_sum_exp

def construct():
    start_time = time.time()
    deadline = start_time + 120.0
    safety_margin = 5.0
    effective_deadline = deadline - safety_margin
    
    # Initial solution: uniform 0.5
    h_uniform = np.full(N, 0.5)
    best_h = h_uniform.copy()
    best_score = _max_overlap(best_h)
    
    # Generate diverse initial points
    def random_initial():
        h = np.random.uniform(0.0, 1.0, size=N)
        h = _fix_sum(h, N/2.0)
        return h
    
    # Also try some structured initial points
    structured_inits = [h_uniform.copy()]
    # Square wave
    for period in [2, 4, 5, 8, 10]:
        sw = np.zeros(N)
        for i in range(N):
            if (i % period) < period // 2:
                sw[i] = 1.0
            else:
                sw[i] = 0.0
        sw = _fix_sum(sw, N/2.0)
        structured_inits.append(sw.copy())
    # High frequency noise
    for freq in [3, 7, 11]:
        t = np.arange(N)
        hf = 0.5 + 0.4 * np.sin(2 * np.pi * freq * t / N)
        hf = np.clip(hf, 0.0, 1.0)
        hf = _fix_sum(hf, N/2.0)
        structured_inits.append(hf.copy())
    # Random inits
    for _ in range(20):
        structured_inits.append(random_initial())
    
    # Simulated annealing with smooth objective
    random.seed(42)
    np.random.seed(42)
    
    def anneal(h_init, start_time, effective_deadline):
        h = h_init.copy()
        # Start with low beta (sharp) and increase (smoother)
        # Actually for minimization of max, we want to start with high beta (smoother) 
        # and decrease to low beta (sharper) to refine
        # Let's do the opposite: start smooth, get global picture, then sharpen
        beta_schedule = [50.0, 30.0, 20.0, 10.0, 5.0, 3.0, 2.0, 1.0, 0.5, 0.25]
        
        best_local = h.copy()
        best_local_score = _max_overlap(h)
        
        for beta in beta_schedule:
            # For each beta, do some L-BFGS-B iterations
            # Use smooth objective with this beta
            def obj_smooth(x):
                if time.time() > effective_deadline:
                    return 1e10
                h_clipped = np.clip(x, 0.0, 1.0)
                # Penalty for sum constraint
                sum_pen = 1000.0 * (_sum_constraint_violation(h_clipped, N/2.0))**2
                return _smooth_max_overlap(h_clipped, beta) + sum_pen
            
            # Use L-BFGS-B with bounds
            try:
                res = minimize(obj_smooth, h, method='L-BFGS-B', 
                              bounds=[(0.0, 1.0)] * N,
                              options={'maxiter': 200, 'gtol': 1e-4})
                if time.time() > effective_deadline:
                    break
                h_new = _fix_sum(res.x, N/2.0)
                score_new = _max_overlap(h_new)
                if score_new < best_local_score:
                    best_local = h_new.copy()
                    best_local_score = score_new
                    h = h_new.copy()
            except Exception:
                pass
        
        # Final polish with low beta / exact objective
        def obj_exact(x):
            if time.time() > effective_deadline:
                return 1e10
            h_clipped = np.clip(x, 0.0, 1.0)
            sum_pen = 1000.0 * (_sum_constraint_violation(h_clipped, N/2.0))**2
            return _max_overlap(h_clipped) + sum_pen
        
        try:
            res = minimize(obj_exact, best_local, method='L-BFGS-B',
                          bounds=[(0.0, 1.0)] * N,
                          options={'maxiter': 500, 'gtol': 1e-6})
            if time.time() <= effective_deadline:
                h_polished = _fix_sum(res.x, N/2.0)
                score_polished = _max_overlap(h_polished)
                if score_polished < best_local_score:
                    best_local = h_polished.copy()
                    best_local_score = score_polished
        except Exception:
            pass
        
        return best_local, best_local_score
    
    # Try each initial point
    for i, h_init in enumerate(structured_inits):
        if time.time() > effective_deadline:
            break
        try:
            h_result, score_result = anneal(h_init, start_time, effective_deadline)
            if score_result < best_score:
                best_h = h_result.copy()
                best_score = score_result
        except Exception:
            continue
    
    # Final check: ensure sum is exactly N/2 and values in [0,1]
    best_h = _fix_sum(best_h, N/2.0)
    best_h = np.clip(best_h, 0.0, 1.0)
    # One more exact sum fix
    diff = N/2.0 - np.sum(best_h)
    if abs(diff) > 1e-12:
        # Distribute the difference evenly? No, that might break [0,1]
        # Just adjust one element
        idx = np.argmin(np.abs(best_h - 0.5))
        best_h[idx] += diff
        best_h = np.clip(best_h, 0.0, 1.0)
        # If clipping broke the sum, try another element
        diff2 = N/2.0 - np.sum(best_h)
        if abs(diff2) > 1e-12:
            # Fallback to uniform
            return h_uniform.tolist()
    
    return best_h.tolist()
# EVOLVE-BLOCK-END
