"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Use 10 localized Gaussian basis functions centered at evenly spaced points
    across [-3, 3] with fixed widths, plus a linear term for global trend.
    
    params layout (10 total):
      params[0] : intercept (constant offset)
      params[1] : linear slope
      params[2:12] : 10 Gaussian weights (but we only have 10 params total)
    
    Wait, we have exactly 10 params. Let's use:
      params[0] : constant
      params[1] : linear coefficient
      params[2:12] : 10 Gaussian weights -> but that's 12 params total.
    
    We need exactly 10 params. So let's use:
      params[0:10] : 10 Gaussian weights centered at 10 evenly spaced points in [-3, 3]
    
    But then we lose the linear term. Alternatively, we can include the linear
    behavior implicitly through the Gaussians if they're wide enough, or we can
    dedicate some params to linear terms.
    
    Better design: Use 8 Gaussian basis functions + 1 constant + 1 linear = 10 params.
    Centers: 8 evenly spaced points in [-3, 3].
    """
    # Fixed centers for 8 Gaussian basis functions, evenly spaced in [-3, 3]
    n_gauss = 8
    centers = np.linspace(-3.0, 3.0, n_gauss)
    # Fixed width (sigma) for each Gaussian; chosen to provide local flexibility
    sigma = 1.2  # moderate width for local flexibility without too much overlap
    
    # params[0] = constant, params[1] = linear slope, params[2:10] = 8 Gaussian weights
    const = params[0]
    slope = params[1]
    weights = params[2:10]
    
    # Compute Gaussian basis values: shape (n_points, n_gauss)
    # x has shape (n_points,)
    x_reshaped = x[:, np.newaxis]  # (n_points, 1)
    centers_reshaped = centers[np.newaxis, :]  # (1, n_gauss)
    
    # Compute squared distances
    diff = x_reshaped - centers_reshaped  # (n_points, n_gauss)
    sq_diff = diff * diff
    
    # Compute Gaussian values: exp(-sq_diff / (2 * sigma^2))
    inv_two_sigma_sq = 1.0 / (2.0 * sigma * sigma)
    gauss_vals = np.exp(-sq_diff * inv_two_sigma_sq)  # (n_points, n_gauss)
    
    # Weighted sum of Gaussians
    gauss_contrib = gauss_vals @ weights  # (n_points,)
    
    # Final prediction
    result = const + slope * x + gauss_contrib
    
    return result
# EVOLVE-BLOCK-END
