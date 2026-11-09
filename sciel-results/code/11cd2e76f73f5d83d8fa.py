"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximate the target function using a truncated Chebyshev series of degree 9.
    The input x is scaled to [-1, 1] to ensure numerical stability.
    Chebyshev polynomials are orthogonal on [-1, 1], making this basis well-conditioned
    for BFGS optimization.
    """
    # Scale input to [-1, 1]
    u = x / 2.0
    
    # Initialize Chebyshev polynomials
    # T_0(u) = 1
    t_0 = np.ones_like(u)
    # T_1(u) = u
    t_1 = u
    
    # Start result with first two terms
    # params is length 10
    result = params[0] * t_0 + params[1] * t_1
    
    # Recursively compute T_k for k=2 to 9
    t_prev = t_0
    t_curr = t_1
    
    for k in range(2, 10):
        # T_k(u) = 2*u*T_{k-1}(u) - T_{k-2}(u)
        t_new = 2.0 * u * t_curr - t_prev
        result = result + params[k] * t_new
        t_prev = t_curr
        t_curr = t_new
        
    return result
# EVOLVE-BLOCK-END
