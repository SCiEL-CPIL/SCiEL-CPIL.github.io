"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Use 8 Gaussian basis functions with varying widths + constant + linear term.
    
    params layout (10 total):
      params[0] : constant
      params[1] : linear slope
      params[2:10] : 8 Gaussian weights
    
    Centers are fixed at 8 evenly spaced points in [-3, 3].
    Widths vary: narrower at edges, wider in middle for better local approximation.
    """
    # Fixed centers for 8 Gaussian basis functions
    n_gauss = 8
    centers = np.linspace(-3.0, 3.0, n_gauss)
    
    # Varying widths: use a combination of narrow and wide Gaussians
    # This gives better local flexibility while maintaining global coverage
    # Widths: alternate between narrow (0.8) and wider (1.5) for diversity
    sigmas = np.array([1.0, 0.8, 1.2, 0.8, 1.2, 0.8, 1.2, 1.0])
    
    # params[0] = constant, params[1] = linear slope, params[2:10] = 8 Gaussian weights
    const = params[0]
    slope = params[1]
    weights = params[2:10]
    
    # Compute Gaussian basis values: shape (n_points, n_gauss)
    x_reshaped = x[:, np.newaxis]  # (n_points, 1)
    centers_reshaped = centers[np.newaxis, :]  # (1, n_gauss)
    sigmas_reshaped = sigmas[np.newaxis, :]  # (1, n_gauss)
    
    # Compute squared distances
    diff = x_reshaped - centers_reshaped  # (n_points, n_gauss)
    sq_diff = diff * diff
    
    # Compute Gaussian values with varying widths: exp(-sq_diff / (2 * sigma_i^2))
    inv_two_sigma_sq = 1.0 / (2.0 * sigmas_reshaped * sigmas_reshaped)  # (1, n_gauss)
    gauss_vals = np.exp(-sq_diff * inv_two_sigma_sq)  # (n_points, n_gauss)
    
    # Weighted sum of Gaussians
    gauss_contrib = gauss_vals @ weights  # (n_points,)
    
    # Final prediction
    result = const + slope * x + gauss_contrib
    
    return result
# EVOLVE-BLOCK-END
