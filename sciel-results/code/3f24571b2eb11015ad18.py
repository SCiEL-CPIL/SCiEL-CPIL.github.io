import numpy as np
import time
from scipy.optimize import minimize

N = 50
DELTA = 0.5 / N

def _score_raw(h):
    """Compute the raw score: L2^2 / (L1 * AutoConvMax)."""
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h**2)
    h_conv = np.convolve(h, h, mode='full')
    autoconv_max = DELTA * np.max(h_conv)
    if autoconv_max == 0:
        return 0.0
    return (l2_sq / (l1 * autoconv_max))

def _objective(x):
    """Minimize the peak autoconvolution."""
    h = np.asarray(x, dtype=np.float64)
    h_conv = np.convolve(h, h, mode='full')
    return np.max(h_conv)

def _construct_via_optimization():
    # Initial guess: Uniform
    h0 = np.ones(N) / N
    
    # Constraints: sum(h) = 1, h >= 0
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}]
    bounds = [(0, 1.0) for _ in range(N)]
    
    # Use SLSQP for convex optimization
    try:
        res = minimize(_objective, h0, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 1000, 'ftol': 1e-12})
        if res.success:
            h_opt = res.x
            # Ensure non-negative and normalized
            h_opt = np.maximum(h_opt, 0)
            h_opt /= np.sum(h_opt)
            return list(h_opt)
    except Exception:
        pass
        
    # Fallback to uniform
    return list(h0)

def construct():
    return _construct_via_optimization()
