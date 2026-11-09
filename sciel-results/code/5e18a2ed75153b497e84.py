"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers the function using a 10-term Chebyshev series expansion.
    The basis functions are Chebyshev polynomials of the first kind, scaled to [-1, 1].
    This provides a well-conditioned linear-in-parameters model that can fit
    smooth non-linear functions on [-2, 2] effectively.
    """
    # Map x from [-2, 2] to t in [-1, 1] for standard Chebyshev polynomials
    t = x / 2.0
    
    # Pre-allocate or compute Chebyshev polynomials T_0 to T_9
    # T_0(t) = 1
    # T_1(t) = t
    # T_k(t) = 2*t*T_{k-1}(t) - T_{k-2}(t)
    
    # We need to be careful with shapes. x is an array.
    # params is an array of length 10.
    
    T = np.zeros((10, len(t)))
    T[0, :] = 1.0
    T[1, :] = t
    
    for k in range(2, 10):
        T[k, :] = 2.0 * t * T[k-1, :] - T[k-2, :]
        
    # Compute the linear combination
    # result[i] = sum_j params[j] * T[j, i]
    # This is effectively T.T @ params
    y = T.T @ params
    
    return y
# EVOLVE-BLOCK-END
