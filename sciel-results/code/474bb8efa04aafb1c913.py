"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Linear combination of RBF basis functions for symbolic regression.
    Uses Gaussian RBFs with fixed centers and bandwidths for flexible
    approximation while maintaining linearity in parameters for BFGS.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    
    # Fixed centers spread across the domain [0.1, 6]
    # Mix of evenly-spaced and clustered centers for local flexibility
    centers = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 5.5])
    
    # Fixed bandwidths: mix of narrow (local) and wide (global)
    # Narrower bandwidths capture local features, wider ones capture trends
    bandwidths = np.array([0.4, 0.6, 0.5, 0.8, 0.7, 1.0, 0.9, 0.6, 0.8, 0.5])
    
    # Compute RBF values: exp(-((x - c)^2) / (2 * sigma^2))
    diff = x[:, np.newaxis] - centers[np.newaxis, :]  # (n, 10)
    sigma2 = bandwidths[np.newaxis, :] ** 2  # (1, 10)
    
    # Compute Gaussian kernels
    rbf_values = np.exp(-0.5 * diff ** 2 / sigma2)  # (n, 10)
    
    # Linear combination of RBF basis functions
    y = rbf_values @ params
    
    return y

# EVOLVE-BLOCK-END
