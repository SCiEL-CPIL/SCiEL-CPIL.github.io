"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # 5 fixed evenly-spaced centers across [-3, 3]
    centers = np.array([-3.0, -1.5, 0.0, 1.5, 3.0])
    
    # Extract 5 weights and 5 log-widths from params
    weights = params[:5]
    log_widths = params[5:10]
    
    # Use exp transformation for widths to ensure positivity
    # Initial param value 1.0 -> exp(1) ≈ 2.718, which is good for domain [-3,3]
    widths_eff = np.exp(log_widths)
    
    # Compute RBF features: exp(-((x - center)^2) / (2 * width^2))
    x_expanded = x[:, np.newaxis]
    centers_expanded = centers[np.newaxis, :]
    
    # Shape: (n_points, 5)
    diff = x_expanded - centers_expanded
    rbf = np.exp(-diff**2 / (2 * widths_eff**2))
    
    # Weighted sum of RBFs
    return rbf @ weights
# EVOLVE-BLOCK-END
