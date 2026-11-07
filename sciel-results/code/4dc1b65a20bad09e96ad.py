"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    10-term Fourier series with fixed frequencies {1, 2, 3, 4, 5} (sines and cosines).
    The input x is scaled to [-pi, pi] to leverage the orthogonality of trigonometric
    functions, creating a well-conditioned linear system for BFGS.
    Basis: sin(u), cos(u), sin(2u), cos(2u), ..., sin(5u), cos(5u) where u = pi*x/3.
    """
    x = np.asarray(x, dtype=float)
    
    # Ensure x is 1D for safety
    if x.ndim != 1:
        x = x.flatten()
        
    # Scale x to [-pi, pi] to ensure orthogonality of sin(k*u) and cos(k*u)
    u = (np.pi / 3.0) * x
    
    n_points = len(u)
    # Initialize the basis matrix
    # We need 10 terms: sin(1u), cos(1u), ..., sin(5u), cos(5u)
    basis = np.zeros((n_points, 10))
    
    # Fill the basis matrix
    # Frequencies 1 to 5
    for k in range(1, 6):
        sin_k = np.sin(k * u)
        cos_k = np.cos(k * u)
        
        # Assign to columns
        # Index mapping:
        # k=1 -> cols 0, 1
        # k=2 -> cols 2, 3
        # ...
        # k=5 -> cols 8, 9
        idx = (k - 1) * 2
        basis[:, idx] = sin_k
        basis[:, idx + 1] = cos_k
        
    # Combine with parameters
    # result = basis @ params
    result = basis @ params
    
    return result
# EVOLVE-BLOCK-END
