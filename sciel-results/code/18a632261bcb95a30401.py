"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Improved basis function using Chebyshev polynomials and smooth non-linearities.
    Mapping x from [-2, 2] to u in [-1, 1] improves conditioning for BFGS.
    """
    x = np.asarray(x, dtype=np.float64)
    p = params
    
    # Map to [-1, 1] for better scaling
    u = x / 2.0
    
    # Chebyshev polynomials T0..T4
    t0 = np.ones_like(u)
    t1 = u
    t2 = 2 * u**2 - 1
    t3 = 4 * u**3 - 3 * u
    t4 = 8 * u**4 - 8 * u**2 + 1
    
    # Smooth non-linear functions
    # sin/cos with period adjusted for u in [-1, 1] -> pi*u covers half period
    s1 = np.sin(np.pi * u)
    c1 = np.cos(np.pi * u)
    
    # Gaussian centered at 0
    g1 = np.exp(-u**2)
    
    # Tanh
    th1 = np.tanh(u)
    
    # Odd Gaussian
    g_odd = u * np.exp(-u**2)
    
    # Build basis matrix (N, 10)
    # Using matrix multiplication for efficiency and clarity
    basis = np.column_stack((
        t0, t1, t2, t3, t4,
        s1, c1, g1, th1, g_odd
    ))
    
    # Linear combination
    result = basis @ p
    
    # Safety check for non-finite values
    if not np.all(np.isfinite(result)):
        return np.zeros_like(x)
        
    return result
# EVOLVE-BLOCK-END
