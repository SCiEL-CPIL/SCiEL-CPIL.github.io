# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _smooth_max(values, T=1.0):
    """Compute a smooth approximation of the maximum."""
    if len(values) == 0:
        return 0.0
    max_val = np.max(values)
    # Avoid overflow in exp
    shifted = values - max_val
    log_sum_exp = np.log(np.sum(np.exp(shifted / T))) + max_val / T
    return T * log_sum_exp


def _objective(x):
    """
    Objective function for L-BFGS-B.
    x: unbounded parameters
    Returns: Upper bound on C3 (to be minimized)
    """
    # Map to [-1, 1]
    h = np.tanh(x)
    
    # Compute autoconvolution
    c = np.convolve(h, h, mode='full')
    
    # The peak is at index N-1.
    # Off-peak values are indices 0 to N-2 and N to 2N-2.
    # Due to symmetry, we only need to consider one half, e.g., N to 2N-2.
    off_peak = c[N:]
    
    # Take absolute values
    abs_off_peak = np.abs(off_peak)
    
    # Smooth max of off-peak values
    T = 0.5
    max_off_peak = _smooth_max(abs_off_peak, T=T)
    
    # L1 norm of h
    l1_norm = np.sum(np.abs(h))
    
    # Avoid division by zero
    if l1_norm < 1e-6:
        return 1e6
        
    value = max_off_peak / (l1_norm ** 2)
    return value


def construct():
    N = 400
    
    # Initialize x with random values to break symmetry and encourage a binary-like solution
    np.random.seed(42)
    x0 = np.random.uniform(-5, 5, N)
    
    # Optimize
    res = minimize(_objective, x0, method='L-BFGS-B', options={'maxiter': 2000, 'ftol': 1e-12})
    
    h_opt = np.tanh(res.x)
    return h_opt
# EVOLVE-BLOCK-END
