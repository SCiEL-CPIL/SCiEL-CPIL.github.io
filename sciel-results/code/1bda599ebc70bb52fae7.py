"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    RBF model with 5 polynomial terms and 5 Gaussian RBF terms.
    
    Design:
    1. Normalize input x from [-3, 3] to z in [-1, 1].
    2. Use 5 polynomial terms: 1, z, z^2, z^3, z^4.
    3. Use 5 Gaussian RBF terms with centers at -1, -0.5, 0, 0.5, 1 (corresponding to -3, -1.5, 0, 1.5, 3 in x-space).
    4. Fixed width sigma = 0.5 in z-space.
    
    params layout:
      params[0:5]  -> coefficients for [1, z, z^2, z^3, z^4]
      params[5:10] -> coefficients for 5 Gaussian kernels
    """
    x = np.asarray(x, dtype=float)
    
    # Normalize input to [-1, 1]
    z = x / 3.0
    
    # Polynomial basis: 1, z, z^2, z^3, z^4
    poly_terms = np.stack([
        np.ones_like(z),
        z,
        z ** 2,
        z ** 3,
        z ** 4
    ], axis=-1)  # shape: (N, 5)
    
    # RBF Gaussian kernels
    # 5 centers corresponding to {-3, -1.5, 0, 1.5, 3} in x-space
    centers = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    sigma = 0.5
    
    # Compute Gaussian kernels: shape (N, 5)
    diff = z[:, np.newaxis] - centers[np.newaxis, :]
    rbf_terms = np.exp(-0.5 * (diff / sigma) ** 2)
    
    # Combine polynomial and RBF terms: shape: (N, 10)
    basis = np.hstack([poly_terms, rbf_terms])
    
    # Prediction: dot product with params
    prediction = basis @ params
    
    return prediction
# EVOLVE-BLOCK-END
