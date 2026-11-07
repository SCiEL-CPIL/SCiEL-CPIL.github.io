"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximates the hidden function using a linear combination of Chebyshev polynomials.
    Input x is mapped from [-2, 2] to [-1, 1] to utilize the standard Chebyshev domain.
    Uses the first 10 Chebyshev polynomials (T_0 to T_9).
    """
    # Ensure x is a numpy array of floats
    x = np.asarray(x, dtype=float)
    
    # Map x from [-2, 2] to t in [-1, 1]
    t = x / 2.0
    
    # Initialize the result as zero
    result = np.zeros_like(t)
    
    # Compute Chebyshev polynomials T_k(t) using the recurrence:
    # T_0(t) = 1
    # T_1(t) = t
    # T_{k+1}(t) = 2*t*T_k(t) - T_{k-1}(t)
    
    # T_0
    T_prev = np.ones_like(t)
    result += params[0] * T_prev
    
    if len(params) > 1:
        # T_1
        T_curr = t
        result += params[1] * T_curr
        
        # Compute T_2 to T_9
        for k in range(2, min(10, len(params))):
            T_next = 2.0 * t * T_curr - T_prev
            result += params[k] * T_next
            T_prev = T_curr
            T_curr = T_next
            
    return result
# EVOLVE-BLOCK-END
