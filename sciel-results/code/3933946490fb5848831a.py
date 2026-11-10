"""Symbolic regression with Gaussian RBF basis."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Model: sum of 5 Gaussian basis functions.
    
    Parameters (10 total):
    - params[0:5]  -> amplitudes a_i
    - params[5:10] -> centers (means) mu_i, scaled to [-2, 2] via tanh
    
    Widths are fixed at sigma = 0.6 for numerical stability with BFGS.
    
    The tanh scaling on centers ensures they stay in a reasonable range
    and the model is smooth in params, aiding BFGS convergence.
    """
    x = np.asarray(x, dtype=np.float64)
    p = params
    
    # Extract amplitudes and scaled centers
    amps = p[0:5]
    # Scale centers to [-2, 2] using tanh to keep them bounded
    centers = 2.0 * np.tanh(p[5:10])
    
    # Fixed width for all Gaussians
    sigma = 0.6
    
    # Compute the 5 Gaussian basis functions
    # Shape: (n_points, 5)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]  # (n, 5)
    gaussians = np.exp(-0.5 * (diff / sigma) ** 2)    # (n, 5)
    
    # Linear combination
    result = gaussians @ amps
    
    return result
# EVOLVE-BLOCK-END
