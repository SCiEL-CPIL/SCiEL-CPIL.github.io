import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a combined basis of Chebyshev polynomials
    and trigonometric functions.
    
    We use 10 parameters:
    - 7 for Chebyshev polynomials T_0 through T_6 (captures smooth polynomial behavior)
    - 3 for trigonometric terms sin(x), cos(x), sin(2x) (captures periodic components)
    
    This hybrid basis is well-conditioned for BFGS and can fit a wider class of
    functions than pure polynomials alone.
    """
    x = np.asarray(x, dtype=np.float64)
    n_points = x.size
    
    # Scale input to [-1, 1] for Chebyshev polynomials
    # Original domain is [-3, 3]
    u = x / 3.0
    
    # Number of Chebyshev basis functions
    n_cheb = 7
    
    # Initialize the basis matrix
    # Shape: (n_points, 10)
    V = np.zeros((n_points, 10), dtype=np.float64)
    
    # Compute Chebyshev polynomials T_0(u) through T_6(u)
    # T_0(u) = 1
    V[:, 0] = 1.0
    
    # T_1(u) = u
    V[:, 1] = u
    
    # Recurrence: T_{k+1}(u) = 2*u*T_k(u) - T_{k-1}(u)
    for k in range(2, n_cheb):
        V[:, k] = 2.0 * u * V[:, k-1] - V[:, k-2]
    
    # Trigonometric terms (indices 7, 8, 9)
    V[:, 7] = np.sin(x)
    V[:, 8] = np.cos(x)
    V[:, 9] = np.sin(2.0 * x)
    
    # Compute the linear combination
    # params shape: (10,)
    # V shape: (n_points, 10)
    # Result shape: (n_points,)
    return V @ params
