"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    RBF + polynomial basis for symbolic regression.
    
    params (length 10):
      params[0]: constant
      params[1]: linear (x)
      params[2]: quadratic (x^2)
      params[3]: cubic (x^3)
      params[4]: quartic (x^4)
      params[5..9]: RBF weights for Gaussian kernels centered at {-3, -1.5, 0, 1.5, 3}
    """
    # Polynomial terms
    y = params[0] + params[1] * x + params[2] * x**2 + params[3] * x**3 + params[4] * x**4
    
    # RBF terms with fixed centers
    centers = np.array([-3.0, -1.5, 0.0, 1.5, 3.0])
    # Fixed width for Gaussian kernels: sigma = 1.5 (reasonable for [-3,3] domain)
    sigma = 1.5
    
    for i in range(5):
        diff = x - centers[i]
        kernel = np.exp(-(diff**2) / (2.0 * sigma**2))
        y += params[5 + i] * kernel
    
    return y
# EVOLVE-BLOCK-END
