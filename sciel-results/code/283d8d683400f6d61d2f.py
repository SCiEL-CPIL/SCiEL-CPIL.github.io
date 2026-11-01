"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _c3_upper_bound(x0):
    """
    Optimizes the signed step function to minimize the L3 norm of the autocorrelation
    normalized by the L1 norm. This corresponds to a tighter 'Third Autocorrelation'
    bound than the peak-based metric used in the seed.
    """
    def objective(x):
        h = np.tanh(x)
        conv = np.convolve(h, h, mode='full')
        # L3 norm of autocorrelation
        # Using abs(conv)**3 to ensure differentiability (C1 smooth)
        sum_cubes = np.sum(np.abs(conv)**3)
        l3_norm = sum_cubes ** (1.0/3.0)
        
        # L1 norm of h
        l1_norm = np.sum(np.abs(h))
        
        # Normalization factor: (L1)^4/3
        if l1_norm < 1e-10:
            return 1e10
        denom = l1_norm ** (4.0/3.0)
        
        return l3_norm / denom

    # Initialize with random signs to avoid the 'all ones' local minimum
    # which has a large low-frequency autocorrelation.
    # Random signs have small sidelobes, reducing the L3 norm significantly.
    np.random.seed(42)
    x0 = np.random.uniform(-10, 10, size=N)
    
    # Use L-BFGS-B for gradient-based optimization
    res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
    
    h_opt = np.tanh(res.x)
    return h_opt


def construct():
    return _c3_upper_bound(np.random.uniform(-10, 10, size=N))
# EVOLVE-BLOCK-END
