"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Radial Basis Function (RBF) model with fixed centers and widths.
    This approach is linear in the parameters, allowing BFGS to fit them efficiently.
    It provides a flexible, non-polynomial basis suitable for arbitrary smooth functions.
    """
    # Define fixed centers for the RBFs
    # Spread 10 centers evenly across the input domain [0.1, 6]
    # Adding a small margin to ensure coverage
    centers = np.linspace(0.2, 5.9, 10)
    
    # Define a fixed width (sigma) for the Gaussian kernels
    # A width of 1.0 provides good overlap between neighbors
    sigma = 1.0
    
    # Compute the RBF basis values
    # x shape: (N,)
    # centers shape: (10,)
    # We want a matrix of shape (N, 10) where basis[i, j] = exp(-(x[i]-centers[j])^2 / (2*sigma^2))
    
    # Efficient vectorized computation
    # diff shape: (N, 10)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    
    # Gaussian kernel
    basis = np.exp(-(diff ** 2) / (2.0 * sigma ** 2))
    
    # Prediction is the linear combination of basis functions
    # params shape: (10,)
    return basis @ params
# EVOLVE-BLOCK-END
