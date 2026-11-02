"""Optimized program for the uncertainty inequality task."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.special import factorial
import warnings

def _compute_bound_analytic_ratios(x4, x8):
    """
    Compute the uncertainty product A(g)A(Fourier(g)) given ratios x4=c4/c0, x8=c8/c0.
    """
    # Constants for A_n^2 / A_0^2
    # A_n^2 = 2^n n! sqrt(pi)
    # A_0^2 = sqrt(pi)
    # Ratio R_n = 2^n n!
    R0 = 1.0
    R4 = 16.0 * 24.0  # 384
    R8 = 256.0 * 40320.0  # 10321920
    R12 = 4096.0 * factorial(12)  # 4096 * 479001600
    
    # D_n = n + 0.5
    D0 = 0.5
    D4 = 4.5
    D8 = 8.5
    D12 = 12.5
    
    # H_n(0)
    H0 = 1.0
    H4 = 12.0
    H8 = 1680.0
    H12 = 665280.0
    
    # c12 ratio
    # c0 + 12 c4 + 1680 c8 + 665280 c12 = 0
    # 1 + 12 x4 + 1680 x8 + 665280 x12 = 0
    x12 = -(1.0 + 12.0 * x4 + 1680.0 * x8) / 665280.0
    
    # Weights w_n proportional to c_n^2 A_n^2
    # w_n = x_n^2 R_n (assuming c0=1, so x0=1)
    w0 = 1.0 * R0
    w4 = x4**2 * R4
    w8 = x8**2 * R8
    w12 = x12**2 * R12
    
    N = w0 + w4 + w8 + w12
    if N < 1e-15:
        return 1e10
        
    M_g = D0 * w0 + D4 * w4 + D8 * w8 + D12 * w12
    
    # A(g)A(Fourier(g)) = M_g / N
    return M_g / N

def _objective_ratios(params):
    x4, x8 = params
    try:
        val = _compute_bound_analytic_ratios(x4, x8)
        if not np.isfinite(val):
            return 1e10
        return val
    except Exception:
        return 1e10

def construct():
    """
    Find coefficients that minimize the upper bound on A(f)*A(Fourier(f)).
    """
    # Initial guess
    # From analysis, x4 = -1/12, x8 = 0 was a candidate.
    # Let's start near there.
    initial = np.array([-1.0/12.0, 0.0])
    
    # Use Nelder-Mead for local optimization
    res = minimize(_objective_ratios, initial, method='Nelder-Mead', options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 10000})
    
    if res.success or res.fun < 1e9:
        x4_opt, x8_opt = res.x
        # Return coefficients c0, c4, c8
        # We can choose c0 = 1.0
        c0 = 1.0
        c4 = x4_opt * c0
        c8 = x8_opt * c0
        return np.array([c0, c4, c8])
    
    # Fallback
    return np.array([1.0, -1.0/12.0, 0.0])
# EVOLVE-BLOCK-END
