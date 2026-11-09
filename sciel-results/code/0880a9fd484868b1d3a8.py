"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid basis: 5 wide Gaussian bumps + 5 narrow Gaussian bumps.
    This provides both global smoothness (wide Gaussians) and local
    flexibility (narrow Gaussians) for capturing sharp features.
    """
    x = np.asarray(x, dtype=np.float64)
    x_expanded = x[:, np.newaxis]
    
    # Wide Gaussians (sigma=1.2) centered at: -2, -1, 0, 1, 2
    wide_centers = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    wide_sigma = 1.2
    
    # Narrow Gaussians (sigma=0.5) centered at: -1.5, -0.5, 0.5, 1.5, 0
    # Note: center 0 is shared with wide, but different width gives local detail
    narrow_centers = np.array([-1.5, -0.5, 0.0, 0.5, 1.5])
    narrow_sigma = 0.5
    
    # Compute wide Gaussian terms
    wide_diff = x_expanded - wide_centers[np.newaxis, :]
    wide_gaussians = np.exp(-0.5 * (wide_diff / wide_sigma) ** 2)
    
    # Compute narrow Gaussian terms
    narrow_diff = x_expanded - narrow_centers[np.newaxis, :]
    narrow_gaussians = np.exp(-0.5 * (narrow_diff / narrow_sigma) ** 2)
    
    # Combine: params[0:5] for wide, params[5:10] for narrow
    result = (wide_gaussians @ params[0:5] + 
              narrow_gaussians @ params[5:10])
    
    return result
# EVOLVE-BLOCK-END
