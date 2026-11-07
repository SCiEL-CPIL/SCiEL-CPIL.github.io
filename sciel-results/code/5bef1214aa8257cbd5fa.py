# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Localized Gaussian basis function approximator.
    
    Uses 10 Gaussian bumps centered at evenly spaced points in [-3, 3].
    Fixed widths for numerical stability.
    
    p[0:10] -> weights for Gaussian basis functions
    """
    p = params
    x = np.asarray(x, dtype=np.float64)
    
    # Centers evenly spaced from -3 to 3
    n_centers = 10
    centers = np.linspace(-3.0, 3.0, n_centers)
    
    # Fixed width for all Gaussians - wide enough for smooth coverage
    # Width of 1.0 gives good balance between local flexibility and smoothness
    sigma = 1.0
    
    # Compute all basis functions
    result = np.zeros_like(x)
    for i in range(n_centers):
        # Gaussian: exp(-0.5 * ((x - center) / sigma)^2)
        diff = (x - centers[i]) / sigma
        basis = np.exp(-0.5 * diff * diff)
        result += p[i] * basis
    
    return result
# EVOLVE-BLOCK-END
