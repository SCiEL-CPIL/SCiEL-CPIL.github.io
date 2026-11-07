"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    10-parameter Gaussian RBF network with fixed centers and widths.
    Centers are evenly spaced in [-2, 2].
    Widths are fixed to a reasonable value.
    Only the 10 weights are optimized (convex problem).
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    params = np.asarray(params, dtype=np.float64).ravel()
    
    # 10 fixed centers evenly spaced in [-2, 2]
    centers = np.linspace(-2.0, 2.0, 10)
    
    # Fixed width: distance between adjacent centers is 4/9 ≈ 0.444
    # Use a width that gives good coverage: sigma = 0.5
    sigma = 0.5
    
    # Compute Gaussian RBFs
    # x has shape (n,), centers has shape (10,)
    # diff has shape (n, 10)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    # squared distances
    sq_dist = diff ** 2
    # Gaussian: exp(-sq_dist / (2 * sigma^2))
    inv_2sigma2 = 1.0 / (2.0 * sigma * sigma)
    rbf = np.exp(-sq_dist * inv_2sigma2)  # shape (n, 10)
    
    # Linear combination with the 10 parameters
    # rbf @ params gives shape (n,)
    prediction = rbf @ params
    
    return prediction
# EVOLVE-BLOCK-END
