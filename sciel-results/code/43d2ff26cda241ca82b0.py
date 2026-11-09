"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers the unknown function using a scaled Chebyshev polynomial basis of degree 9.
    Each basis function is normalized to have unit L2 norm over the training domain,
    which improves conditioning for BFGS optimization.
    """
    x_flat = np.asarray(x).flatten()
    n_points = x_flat.shape[0]
    
    if n_points == 0:
        return np.array([])
        
    # Map input x from [-3, 3] to [-1, 1] for stable Chebyshev evaluation
    t = x_flat / 3.0
    
    # Precompute Chebyshev polynomials T_0 to T_9
    T = np.zeros((10, n_points))
    T[0, :] = 1.0
    T[1, :] = t
    for k in range(2, 10):
        T[k, :] = 2.0 * t * T[k-1, :] - T[k-2, :]
    
    # Normalize each basis function to have unit L2 norm
    # This makes the parameter scaling more natural for BFGS
    norms = np.sqrt(np.sum(T ** 2, axis=1))
    norms[norms < 1e-12] = 1.0  # Avoid division by zero
    T = T / norms[:, np.newaxis]
    
    # Linear combination: y = sum(params[i] * T[i, :])
    y = np.dot(params, T)
    
    return y
# EVOLVE-BLOCK-END
