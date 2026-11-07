"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximates the target function using a Chebyshev series of degree 9.
    The input x is scaled to [-1, 1] to improve numerical stability and
    utilize the standard Chebyshev polynomial definition.
    
    params: array of 10 coefficients [c0, c1, ..., c9]
    """
    # Ensure x is 1D for consistent broadcasting
    x = np.asarray(x).flatten()
    
    # Scale x from [-2, 2] to [-1, 1]
    u = x / 2.0
    
    # Pre-allocate array for Chebyshev polynomial values
    # Shape: (N, 10) where N is number of points
    T = np.empty((len(x), 10), dtype=np.float64)
    
    # T_0(u) = 1
    T[:, 0] = 1.0
    
    # T_1(u) = u
    T[:, 1] = u
    
    # Recurrence relation: T_{n+1}(u) = 2*u*T_n(u) - T_{n-1}(u)
    for n in range(2, 10):
        T[:, n] = 2.0 * u * T[:, n-1] - T[:, n-2]
        
    # Compute the linear combination: y = sum(c_i * T_i(u))
    # T is (N, 10), params is (10,)
    # Result is (N,)
    y_pred = T @ params
    
    return y_pred
# EVOLVE-BLOCK-END
