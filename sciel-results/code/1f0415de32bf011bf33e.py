"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Constructs a prediction using a fixed-center Radial Basis Function (RBF) network.
    The centers are evenly spaced across the input domain [0.1, 6.0].
    The parameters p0..p9 control the amplitude of each Gaussian bump.
    This allows the model to approximate local features of the hidden function
    without relying on a specific global functional form.
    """
    # Ensure x is a numpy array for safe operations
    x = np.asarray(x, dtype=np.float64)
    
    # Define 10 fixed centers evenly spaced in [0.1, 6.0]
    centers = np.linspace(0.1, 6.0, 10)
    
    # Define a fixed width for the Gaussians.
    # A width of 0.5 provides sufficient overlap to ensure smoothness
    # while maintaining local sensitivity.
    sigma = 0.5
    
    # Precompute the squared differences for all x and centers
    # x has shape (N,), centers has shape (10,)
    # diff has shape (N, 10)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    
    # Compute Gaussian basis functions
    # b has shape (N, 10)
    b = np.exp(-(diff ** 2) / (2 * sigma ** 2))
    
    # Linear combination of basis functions using params
    # params has shape (10,)
    # y = b @ params
    y = b @ params
    
    return y
# EVOLVE-BLOCK-END
