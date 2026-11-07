"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Flatten input to ensure 1D processing
    x_flat = np.asarray(x).flatten()
    n_points = x_flat.shape[0]
    
    # Map x from [-3, 3] to t in [-1, 1] for Legendre polynomials
    t = x_flat / 3.0
    
    # Precompute Legendre polynomials P_0 to P_9
    # P is a matrix of shape (n_points, 10)
    P = np.zeros((n_points, 10), dtype=np.float64)
    
    if n_points == 0:
        return np.array([])
        
    P[:, 0] = 1.0
    P[:, 1] = t
    
    for n in range(2, 10):
        # Recurrence: P_n(t) = ((2n - 1) * t * P_{n-1}(t) - (n - 1) * P_{n-2}(t)) / n
        P[:, n] = ((2.0 * n - 1.0) * t * P[:, n-1] - (n - 1.0) * P[:, n-2]) / n
        
    # Linear combination of basis functions
    # params is expected to be of length 10
    # Ensure params is 1D
    p = np.asarray(params).flatten()
    
    # If params length is not 10, pad or truncate? 
    # The task says "fits params, an array of length 10". 
    # So we can assume len(params) == 10.
    
    return P @ p
# EVOLVE-BLOCK-END
