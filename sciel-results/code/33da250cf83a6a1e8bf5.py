# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form prediction using Gaussian Radial Basis Functions (RBFs) with 10 fixed centers
    evenly spaced across [0.1, 6]. This creates a smooth function class that can approximate
    a wide variety of continuous functions. Linear in parameters.
    """
    x = np.asarray(x, dtype=np.float64)
    
    if not np.all(np.isfinite(x)):
        return np.zeros_like(x)
    
    p = np.asarray(params, dtype=np.float64)
    
    # Define 10 evenly spaced centers across [0.1, 6]
    x_min, x_max = 0.1, 6.0
    centers = np.linspace(x_min, x_max, 10)
    
    # Fixed width for the Gaussian RBFs
    # Spacing is approx (6.0 - 0.1) / 9 = 0.655
    # Choose sigma to be about half the spacing for good local coverage
    sigma = 0.35
    
    pred = np.zeros_like(x)
    
    for i in range(10):
        # Gaussian RBF: exp(-(x - c_i)^2 / (2 * sigma^2))
        diff = x - centers[i]
        rbf = np.exp(-(diff * diff) / (2.0 * sigma * sigma))
        pred += p[i] * rbf
    
    # Ensure finite output
    pred = np.where(np.isfinite(pred), pred, 0.0)
    
    return pred
# EVOLVE-BLOCK-END
