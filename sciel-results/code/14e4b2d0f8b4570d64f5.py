"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers the unknown function using a Chebyshev polynomial basis of degree 9.
    This provides a stable, flexible model for smooth functions on [-3, 3].
    """
    # Ensure x is a 1D array for consistent shape handling
    x_flat = np.asarray(x).flatten()
    n_points = x_flat.shape[0]
    
    if n_points == 0:
        return np.array([])
        
    # Map input x from [-3, 3] to [-1, 1] for stable Chebyshev evaluation
    t = x_flat / 3.0
    
    # Precompute Chebyshev polynomials T_0 to T_9
    # T_0 = 1
    # T_1 = t
    # T_k = 2*t*T_{k-1} - T_{k-2}
    
    T = np.zeros((10, n_points))
    T[0, :] = 1.0
    T[1, :] = t
    for k in range(2, 10):
        T[k, :] = 2.0 * t * T[k-1, :] - T[k-2, :]
        
    # Normalize basis functions to have unit L2 norm on [-1, 1]
    # This improves the conditioning of the linear system for BFGS.
    # ||T_0||^2 = 2, ||T_k||^2 = pi/2 for k >= 1
    T[0, :] /= np.sqrt(2.0)
    for k in range(1, 10):
        T[k, :] /= np.sqrt(np.pi / 2.0)

    # Linear combination: y = sum(params[i] * T[i, :])
    # params is shape (10,), T is shape (10, N)
    # Result is shape (N,)
    y = np.dot(params, T)
    
    return y
# EVOLVE-BLOCK-END
