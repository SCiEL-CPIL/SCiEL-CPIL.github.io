"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    RBF network with 10 fixed Gaussian centers spanning [0.1, 6].
    params[0:10]: weights for each of the 10 RBF centers.
    
    The centers are fixed at equally spaced points in the domain.
    The width (sigma) is fixed to a value that ensures sufficient overlap
    between neighboring kernels, providing a smooth basis for the linear combination.
    This formulation results in a convex optimization problem for the weights,
    allowing BFGS to converge reliably to the optimal linear combination.
    """
    # Fixed centers spanning the input domain [0.1, 6]
    # Using 10 centers to match the 10 available parameters for weights
    centers = np.linspace(0.1, 6.0, 10)
    
    # Fixed width
    # Domain size is ~5.9, 10 centers -> spacing ~0.65
    # Sigma = 0.5 provides good overlap (neighbor distance is ~1 sigma away)
    sigma = 0.5
    
    # Compute distances from x to each center
    # x shape: (N,)
    # centers shape: (10,)
    # diff shape: (N, 10)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    
    # Compute Gaussian kernels
    # exp(-diff^2 / (2 * sigma^2))
    kernel = np.exp(-diff**2 / (2.0 * sigma**2))
    
    # Linear combination of kernels
    # weights shape: (10,)
    weights = params
    
    # prediction = kernel @ weights
    prediction = kernel @ weights
    
    return prediction
# EVOLVE-BLOCK-END
