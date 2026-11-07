"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Single hidden-layer RBF network with 10 parameters.
    params[0]: constant bias
    params[1:10]: weights for 9 Gaussian RBF units
    
    RBF centers are fixed at evenly spaced points in [0.1, 6].
    Width is fixed to provide good coverage of the input range.
    """
    # 9 centers evenly spaced in [0.1, 6]
    centers = np.linspace(0.1, 6.0, 9)
    # Fixed width - chosen to cover the range with some overlap
    width = 1.2
    
    # Build RBF matrix: shape (n_points, 9)
    x_expanded = x[:, np.newaxis]  # (n, 1)
    centers_expanded = centers[np.newaxis, :]  # (1, 9)
    
    # Compute squared distances
    dist_sq = (x_expanded - centers_expanded) ** 2  # (n, 9)
    
    # RBF values: exp(-dist_sq / (2 * width^2))
    rbf_values = np.exp(-dist_sq / (2.0 * width ** 2))  # (n, 9)
    
    # Combine: constant + weighted sum of RBFs
    # params[0] is constant, params[1:10] are RBF weights
    result = params[0] + rbf_values @ params[1:10]
    
    return result
# EVOLVE-BLOCK-END
