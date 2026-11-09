"""Symbolic regression with 3 inputs using a Radial Basis Function (RBF) basis."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression using a Radial Basis Function (RBF) basis.
    
    The model is:
    f(x, y, z) = p[0] + sum_{i=1}^{9} p[i] * exp(-||r - c_i||^2 / (2 * sigma^2))
    
    Where:
    - p[0] is a constant offset.
    - p[1..9] are amplitudes for 9 Gaussian RBFs.
    - c_i are fixed centers in the domain [0.5, 2]^3.
    - sigma is a fixed width parameter.
    
    This approach differs from the polynomial basis by using localized non-linear
    features, allowing it to capture complex spatial variations that polynomials
    might miss or require high degrees to approximate.
    """
    p = params
    
    if not np.all(np.isfinite(p)):
        return np.ones_like(x)
    
    # Fixed width for the Gaussians
    sigma = 0.8
    sigma_sq_2 = 2 * sigma * sigma
    
    # Define 9 centers spread across the domain [0.5, 2.0]
    # We use a grid of points to ensure coverage.
    # Centers:
    # (0.5, 0.5, 0.5), (2.0, 0.5, 0.5), (0.5, 2.0, 0.5), (2.0, 2.0, 0.5)
    # (0.5, 0.5, 2.0), (2.0, 0.5, 2.0), (0.5, 2.0, 2.0), (2.0, 2.0, 2.0)
    # (1.25, 1.25, 1.25) -- Center
    
    cx = np.array([0.5, 2.0, 0.5, 2.0, 0.5, 2.0, 0.5, 2.0, 1.25])
    cy = np.array([0.5, 0.5, 2.0, 2.0, 0.5, 0.5, 2.0, 2.0, 1.25])
    cz = np.array([0.5, 0.5, 0.5, 0.5, 2.0, 2.0, 2.0, 2.0, 1.25])
    
    # Calculate squared distances from each point (x, y, z) to each center
    # x, y, z are 1D arrays of shape (N,)
    # cx, cy, cz are 1D arrays of shape (9,)
    
    # To avoid memory issues with large N, we compute in chunks or use broadcasting carefully.
    # N is 200, so N*9 is small. Broadcasting is fine.
    
    dx = x[:, np.newaxis] - cx[np.newaxis, :]  # Shape (N, 9)
    dy = y[:, np.newaxis] - cy[np.newaxis, :]  # Shape (N, 9)
    dz = z[:, np.newaxis] - cz[np.newaxis, :]  # Shape (N, 9)
    
    dist_sq = dx**2 + dy**2 + dz**2  # Shape (N, 9)
    
    # Compute Gaussian values
    gaussians = np.exp(-dist_sq / sigma_sq_2)  # Shape (N, 9)
    
    # Linear combination with parameters p[1:10]
    # p[0] is the constant term
    result = p[0] + gaussians @ p[1:10]
    
    return result
# EVOLVE-BLOCK-END
