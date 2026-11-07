"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    p = params
    
    # Ensure we handle the fixed size of params (10)
    if len(p) < 10:
        p = np.pad(p, (0, 10 - len(p)))
    else:
        p = p[:10]

    # Scale input to [-1, 1]
    u = x / 3.0
    
    # Basis functions: Radial Basis Functions (Gaussians)
    # Centers spread evenly across [-1, 1]
    # This allows approximation of non-periodic smooth functions.
    centers = np.linspace(-1.0, 1.0, 10)
    sigma = 0.4  # Width of the Gaussian bumps
    
    # Compute the basis matrix efficiently
    # u shape: (N,)
    # centers shape: (10,)
    # diff shape: (N, 10)
    diff = u[:, np.newaxis] - centers[np.newaxis, :]
    
    # Basis values: (N, 10)
    basis = np.exp(-0.5 * (diff / sigma) ** 2)
    
    # Return weighted sum: (N, 10) dot (10,) -> (N,)
    return basis @ p
# EVOLVE-BLOCK-END
