"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # 10 localized Gaussian basis functions centered evenly across [-3, 3]
    # Fixed widths chosen to give good local coverage without ill-conditioning
    centers = np.linspace(-3.0, 3.0, 10)
    widths = np.full(10, 0.8)  # fixed widths for stability
    
    # Compute Gaussian basis: exp(-((x - c_i)^2) / (2 * w_i^2))
    # Shape: (n_points, 10)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    basis = np.exp(-0.5 * (diff / widths[np.newaxis, :]) ** 2)
    
    # Linear combination with learned params
    return basis @ params
# EVOLVE-BLOCK-END
