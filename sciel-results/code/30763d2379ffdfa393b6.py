"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import random

def _smooth_abs(x, eps=1e-8):
    return np.sqrt(x**2 + eps)

def _soft_max(x, T=1.0):
    # Stable log-sum-exp
    max_x = np.max(x)
    return T * (np.log(np.sum(np.exp((x - max_x) / T))) + max_x / T)

def _objective(x, T=1.0, eps=1e-8):
    h = np.tanh(x)
    conv = np.convolve(h, h, mode='full')
    abs_conv = _smooth_abs(conv, eps)
    log_max = _soft_max(abs_conv, T)
    sum_abs_h = np.sum(_smooth_abs(h, eps))
    # Avoid division by zero if h is near zero
    denom = sum_abs_h**2 + 1e-12
    return log_max / denom

def _optimize_once(x0, T=1.0):
    # Use L-BFGS-B
    res = minimize(_objective, x0, method='L-BFGS-B', args=(T,), options={'maxiter': 200, 'ftol': 1e-10})
    return res.fun, np.tanh(res.x)

def _c3_upper_bound(x0):
    """
    Compute the upper bound on C3 for a signed step function with heights h.
    Uses gradient-based optimization to find a good h.
    """
    best_val = np.inf
    best_h = np.ones(N)
    
    # Run multiple restarts
    num_restarts = 5
    for i in range(num_restarts):
        if i == 0:
            # Start from the seed-like state (all positive)
            curr_x0 = np.full(N, 5.0)
        else:
            # Random restarts
            curr_x0 = np.random.uniform(-5, 5, N)
        
        try:
            val, h_opt = _optimize_once(curr_x0, T=1.0)
            if val < best_val:
                best_val = val
                best_h = h_opt
        except Exception:
            continue
            
    return best_h

def construct():
    # The seed program passed np.full(N, 5.0) to _c3_upper_bound.
    # We maintain the interface.
    return _c3_upper_bound(np.full(N, 5.0))
# EVOLVE-BLOCK-END
