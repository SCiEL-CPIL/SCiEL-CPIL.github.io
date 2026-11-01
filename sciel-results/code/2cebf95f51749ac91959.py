"""Rewritten program for the uncertainty inequality tuning task."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from math import factorial

_DEADLINE = time.time() + 200


def _compute_variance(c0, c4, c8, c12):
    """Compute <x^2> for the given Hermite coefficients."""
    w0 = 1
    w4 = 16 * 24  # 2^4 * 4!
    w8 = 256 * 40320  # 2^8 * 8!
    w12 = 4096 * factorial(12)  # 2^12 * 12!
    
    num = (c0**2 * w0 * 1 +
           c4**2 * w4 * 9 +
           c8**2 * w8 * 17 +
           c12**2 * w12 * 25)
    
    den = 2.0 * (c0**2 * w0 +
                 c4**2 * w4 +
                 c8**2 * w8 +
                 c12**2 * w12)
    
    if den <= 0:
        return 1e10
    return num / den


def _objective(coeffs):
    c0, c4, c8 = coeffs
    c12 = (c0 - 12.0 * c4 + 1008.0 * c8) / 102816.0
    return _compute_variance(c0, c4, c8, c12)


def construct():
    """Return three coefficients for H0, H4, H8."""
    best_val = 1e10
    best_coeffs = None
    
    start_points = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, -1.0],
        [1.0, -1.0, -1.0],
        [0.5, 0.5, 0.5],
        [1.0, 0.1, 0.01],
        [1.0, -0.1, 0.01],
        [1.0, 0.5, 0.1],
        [1.0, -0.5, 0.1],
        [0.1, 1.0, 0.1],
        [0.1, -1.0, 0.1],
        [1.0, 1.0, 0.0],
        [1.0, -1.0, 0.0],
        [1.0, 0.0, 1.0],
        [1.0, 0.0, -1.0],
    ]
    
    for sp in start_points:
        if time.time() > _DEADLINE:
            break
        try:
            res = minimize(_objective, sp, method='Nelder-Mead',
                          options={'maxiter': 50000, 'xatol': 1e-14, 'fatol': 1e-18, 'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_coeffs = res.x
        except:
            pass
    
    if best_coeffs is None:
        return np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    
    return np.array(best_coeffs)
# EVOLVE-BLOCK-END
