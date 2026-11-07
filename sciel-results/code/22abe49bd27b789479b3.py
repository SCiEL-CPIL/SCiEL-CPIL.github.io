"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid model: Fixed-center Gaussian RBFs + Polynomial basis.
    
    params[0:5]  : Amplitudes for 5 fixed-center Gaussian kernels
    params[5:10] : Coefficients for a 4th-order polynomial basis [1, x, x^2, x^3, x^4]
    
    This design stabilizes optimization by fixing kernel centers, while the
    polynomial term captures global trends and the RBFs capture local features.
    """
    # Fixed centers spread across the domain [-2, 2]
    centers = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    
    # Extract amplitudes and polynomial coefficients
    amplitudes = params[0:5]
    poly_coeffs = params[5:10]
    
    # Reshape for broadcasting
    x_col = x.reshape(-1, 1)
    c_row = centers.reshape(1, -1)
    
    # Compute RBF part
    # Using sigma = 0.8, so 2 * sigma^2 = 1.28
    sq_dist = (x_col - c_row) ** 2
    kernels = np.exp(-sq_dist / 1.28)
    rbf_part = kernels @ amplitudes
    
    # Compute Polynomial part
    # Basis: 1, x, x^2, x^3, x^4
    # x is (N, 1), so x**k is (N, 1)
    # We need to construct (N, 5) matrix for dot product with (5,)
    x_flat = x.flatten()
    poly_basis = np.column_stack([
        np.ones_like(x_flat),
        x_flat,
        x_flat**2,
        x_flat**3,
        x_flat**4
    ])
    poly_part = poly_basis @ poly_coeffs
    
    return rbf_part + poly_part
# EVOLVE-BLOCK-END
