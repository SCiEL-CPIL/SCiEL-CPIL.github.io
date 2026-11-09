"""Rewritten program for symbolic regression with 3 inputs."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression using Radial Basis Functions (RBFs) to approximate
    the hidden function.
    
    We use 9 Gaussian RBFs centered at the corners and center of the domain [0.5, 2]^3,
    plus a constant bias term. This provides a flexible, smooth approximation
    capability that can fit non-polynomial functions better than low-order polynomials.
    
    params has length 10:
    params[0]: bias
    params[1:10]: weights for the 9 RBF centers
    """
    # Define 9 centers: 8 corners + 1 center
    # Domain is [0.5, 2.0]
    # Corners:
    # (0.5, 0.5, 0.5), (0.5, 0.5, 2.0), (0.5, 2.0, 0.5), (0.5, 2.0, 2.0)
    # (2.0, 0.5, 0.5), (2.0, 0.5, 2.0), (2.0, 2.0, 0.5), (2.0, 2.0, 2.0)
    # Center:
    # (1.25, 1.25, 1.25)
    
    centers = np.array([
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 2.0],
        [0.5, 2.0, 0.5],
        [0.5, 2.0, 2.0],
        [2.0, 0.5, 0.5],
        [2.0, 0.5, 2.0],
        [2.0, 2.0, 0.5],
        [2.0, 2.0, 2.0],
        [1.25, 1.25, 1.25]
    ])
    
    # Stack inputs into shape (N, 3)
    inputs = np.column_stack((x, y, z))
    
    # Compute squared distances from each input to each center
    # inputs: (N, 3), centers: (9, 3)
    # dist_sq: (N, 9)
    # Using broadcasting: (N, 1, 3) - (1, 9, 3) -> (N, 9, 3) -> sum over axis 2
    diff = inputs[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    
    # Gaussian RBF: exp(-dist_sq / (2 * sigma^2))
    # Choose sigma. Domain width is 1.5.
    # A sigma of 0.6 gives a reasonable spread.
    sigma = 0.6
    sigma_sq_2 = 2 * sigma * sigma
    rbf_values = np.exp(-dist_sq / sigma_sq_2)
    
    # rbf_values shape: (N, 9)
    # params[1:10] are the weights for these 9 RBFs
    weights = params[1:10]
    
    # Linear combination
    pred = np.sum(rbf_values * weights, axis=1) + params[0]
    
    return pred
# EVOLVE-BLOCK-END
