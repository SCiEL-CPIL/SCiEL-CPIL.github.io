"""Symbolic regression with fixed Gaussian RBF basis."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Linear combination of 10 Gaussian RBFs centered evenly in [-2, 2].
    params: array of length 10 (coefficients).
    Returns: array of predictions, one per point in x.
    """
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # Fixed centers evenly spaced in [-2, 2]
    centers = np.linspace(-2.0, 2.0, 10)
    
    # Fixed bandwidth (sigma) for all RBFs
    sigma = 0.6
    
    # Compute distances: x is (N,), centers is (10,)
    # diff shape: (N, 10)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    
    # Gaussian kernel: exp(-0.5 * (diff/sigma)^2)
    # Shape: (N, 10)
    basis = np.exp(-0.5 * (diff / sigma) ** 2)
    
    # Linear combination: (N, 10) @ (10,) -> (N,)
    return basis @ params
# EVOLVE-BLOCK-END
