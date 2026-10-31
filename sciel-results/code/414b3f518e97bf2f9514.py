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

def _construct_h_smooth(c, w):
    """
    Construct h from center c and width w.
    h[i] approximates a step from 0 to 1 at c with width w.
    """
    i = np.arange(N)
    if w < 1e-6:
        w = 1e-6
    h = 0.5 * (1.0 + np.tanh((i - c) / w))
    return h

def _enforce_sum(h):
    """
    Adjust h to have sum exactly N/2, while keeping values in [0,1].
    """
    target = N / 2.0
    current_sum = np.sum(h)
    diff = target - current_sum
    
    for _ in range(100):
        if abs(diff) < 1e-12:
            break
        h = h + diff / N
        h = np.clip(h, 0.0, 1.0)
        current_sum = np.sum(h)
        diff = target - current_sum
        
    return h

def _objective_1d(log_w):
    """
    Objective function for 1D optimization of log(w).
    c is fixed to N/2 due to circular symmetry.
    """
    w = np.exp(log_w)
    
    # Clamp w
    if w < 1e-3:
        w = 1e-3
    if w > N:
        w = N
        
    c = N / 2.0
    h = _construct_h_smooth(c, w)
    h = _enforce_sum(h)
    
    ov = _max_overlap(h)
    return ov

def construct():
    """
    Construct the sequence of step heights.
    """
    # Optimize log_w
    # Range: log(1e-3) ~ -6.9 to log(100) ~ 4.6
    # Let's search [-6, 5]
    
    # Coarse grid to find good region
    log_w_grid = np.linspace(-6.0, 5.0, 50)
    best_score = float('inf')
    best_log_w = 0.0
    
    for lw in log_w_grid:
        score = _objective_1d(lw)
        if score < best_score:
            best_score = score
            best_log_w = lw
            
    # Local refinement
    # Bounds around best_log_w
    lower = max(-6.0, best_log_w - 1.0)
    upper = min(5.0, best_log_w + 1.0)
    
    res = minimize_scalar(_objective_1d, bounds=(lower, upper), method='bounded', options={'xatol': 1e-6})
    
    if res.fun < best_score:
        best_score = res.fun
        best_log_w = res.x
        
    # Final construction
    w = np.exp(best_log_w)
    c = N / 2.0
    h = _construct_h_smooth(c, w)
    h = _enforce_sum(h)
    
    return h
