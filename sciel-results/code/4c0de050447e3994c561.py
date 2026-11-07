"""Symbolic regression using RBF + polynomial basis."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximates the target function using a hybrid RBF + polynomial basis.
    
    Architecture:
    - 5 Gaussian RBF terms centered at {-3, -1.5, 0, 1.5, 3} with fixed width
    - 5 polynomial terms: [1, x, x^2, x^3, x^4]
    
    Parameters (length 10):
    - params[0:5]: Coefficients for the 5 RBF terms
    - params[5:10]: Coefficients for the 5 polynomial terms
    
    The RBF terms provide local flexibility to capture bumps and non-smooth features,
    while the polynomial terms capture global trends. This localized representation
    handles non-periodic, smooth targets more effectively than global bases.
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Fixed RBF centers
    centers = np.array([-3.0, -1.5, 0.0, 1.5, 3.0])
    # Fixed RBF width (sigma)
    sigma = 1.0
    
    # Compute RBF features: (N, 5)
    # x[:, np.newaxis] - centers[np.newaxis, :] -> (N, 5)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    rbf = np.exp(-diff**2 / (2 * sigma**2))
    
    # Compute polynomial features: (N, 5)
    poly = np.stack([
        np.ones_like(x),
        x,
        x**2,
        x**3,
        x**4
    ], axis=1)
    
    # Unpack parameters
    rbf_coeffs = params[0:5]   # (5,)
    poly_coeffs = params[5:10] # (5,)
    
    # Compute output
    y = rbf @ rbf_coeffs + poly @ poly_coeffs
    
    return y
# EVOLVE-BLOCK-END
