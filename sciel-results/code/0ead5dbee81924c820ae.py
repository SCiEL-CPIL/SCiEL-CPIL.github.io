"""Rewritten program for Erdos minimum overlap task."""
import numpy as np
from scipy.optimize import minimize_scalar

N = 100

def _max_overlap(h):
    """Compute the maximum overlap of h with shifted 1-h over all shifts."""
    one_minus_h = 1.0 - h
    max_ov = 0.0
    for s in range(N):
        shifted = np.roll(one_minus_h, s)
        ov = np.dot(h, shifted)
        if ov > max_ov:
            max_ov = ov
    return max_ov

def _construct_h(w, c):
    """
    Construct h given width w and center c.
    h[i] = 0.5 + 0.5 * tanh((i - c) / w)
    """
    i = np.arange(N)
    # Avoid division by zero
    if w < 1e-6:
        w = 1e-6
    h = 0.5 + 0.5 * np.tanh((i - c) / w)
    return h

def _solve_c(w):
    """
    Find c such that sum(h) = N/2.
    sum(h) is monotonic in c.
    """
    target_sum = N / 2.0
    
    # Binary search for c in [0, N]
    lo = 0.0
    hi = float(N)
    
    for _ in range(50):
        mid = (lo + hi) / 2.0
        h_mid = _construct_h(w, mid)
        s_mid = np.sum(h_mid)
        if s_mid > target_sum:
            lo = mid
        else:
            hi = mid
            
    c_opt = (lo + hi) / 2.0
    return c_opt

def _objective_1d(w_log):
    """
    Objective function for 1D optimization of log(w).
    """
    w = np.exp(w_log)
    # Clamp w to reasonable range
    if w < 1e-3:
        w = 1e-3
    if w > 10000:
        w = 10000
        
    c = _solve_c(w)
    h = _construct_h(w, c)
    
    # Ensure sum is exactly N/2
    current_sum = np.sum(h)
    if abs(current_sum - N/2.0) > 1e-9:
        diff = N/2.0 - current_sum
        h = h + diff / N
        h = np.clip(h, 0.0, 1.0)
        
    return _max_overlap(h)

def construct():
    """
    Construct the sequence of step heights.
    """
    # Theoretical minimum is achieved when h[i] = 0.5 for all i.
    # This corresponds to w -> infinity.
    # We will optimize w but also compare against the flat solution.
    
    h_flat = np.full(N, 0.5)
    score_flat = _max_overlap(h_flat)
    
    # Optimize log(w)
    # Expand bounds to allow larger w
    res = minimize_scalar(_objective_1d, bounds=(0.0, np.log(10000)), method='bounded', options={'xatol': 1e-4})
    
    best_w_log = res.x
    best_w = np.exp(best_w_log)
    
    c = _solve_c(best_w)
    h_opt = _construct_h(best_w, c)
    
    # Final correction for exact sum
    current_sum = np.sum(h_opt)
    diff = N / 2.0 - current_sum
    
    if abs(diff) > 1e-10:
        h_opt = h_opt + diff / N
        h_opt = np.clip(h_opt, 0.0, 1.0)
        
        # Iterative fix for exact sum
        for _ in range(20):
            current_sum = np.sum(h_opt)
            diff = N / 2.0 - current_sum
            if abs(diff) < 1e-12:
                break
            if diff > 0:
                mask = h_opt < 1.0
                if not np.any(mask): break
                indices = np.where(mask)[0]
                step = diff / len(indices)
                for idx in indices:
                    h_opt[idx] = min(1.0, h_opt[idx] + step)
            else:
                mask = h_opt > 0.0
                if not np.any(mask): break
                indices = np.where(mask)[0]
                step = diff / len(indices)
                for idx in indices:
                    h_opt[idx] = max(0.0, h_opt[idx] + step)

    score_opt = _max_overlap(h_opt)
    
    if score_opt < score_flat:
        return h_opt
    else:
        return h_flat

# EVOLVE-BLOCK-START
# (The construct function is defined above, but we need to keep the interface)
# The task says "Keep the function names and signatures of the seed program."
# The seed program has construct() -> sequence of step heights.
# So we just need to make sure construct() is defined and returns the right thing.
# EVOLVE-BLOCK-END
