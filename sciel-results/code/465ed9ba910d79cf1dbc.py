"""Symbolic regression with 1 input(s) using RBF basis."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers the unknown function using a linear combination of Radial Basis Functions (RBFs).
    This provides local flexibility to capture non-polynomial features.
    """
    x = np.asarray(x, dtype=np.float64)
    
    if not np.all(np.isfinite(x)):
        return np.zeros_like(x)
        
    n_points = x.shape[0]
    if n_points == 0:
        return np.array([])
        
    # Define 10 RBF centers evenly spaced in the domain [0.1, 6.0]
    # We use the full range to ensure coverage
    centers = np.linspace(0.1, 6.0, 10)
    
    # Define bandwidths (sigma). 
    # A wider sigma allows for more global influence, narrower for local.
    # sigma=1.5 provides good overlap.
    sigma = 1.5
    
    # Compute RBF values: phi_i(x) = exp(-((x - c_i)^2) / (2 * sigma^2))
    # x is shape (N,), centers is shape (10,)
    # We want basis of shape (N, 10)
    
    # Reshape for broadcasting: x[:, None] is (N, 1), centers[None, :] is (1, 10)
    diff = x[:, None] - centers[None, :]
    
    # Calculate squared distances
    dist_sq = diff * diff
    
    # Calculate RBF values
    basis = np.exp(-dist_sq / (2.0 * sigma * sigma))
    
    # Linear combination
    # params is shape (10,)
    pred = basis @ params
    
    return pred
# EVOLVE-BLOCK-END
