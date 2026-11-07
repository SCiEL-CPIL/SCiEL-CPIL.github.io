"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers the function using a mixed polynomial and reciprocal basis.
    
    Rationale:
    - Previous attempts with pure polynomials (linear, quadratic) and pure exponentials 
      performed poorly (scores ~1.0 to ~7.4).
    - The score of 1.0 for linear suggests the model is just predicting the mean, implying 
      the function has significant non-linear structure that linear terms can't capture.
    - The jump to 7.4 for quadratic suggests either overfitting or that the true function 
      is not well-approximated by low-degree polynomials in the standard basis.
    - Reciprocal terms (1/x, 1/y, 1/z) are common in rational functions and can capture 
      hyperbolic-like behaviors or singularities that polynomials miss.
    - Combining linear terms with reciprocal terms provides a richer basis that can 
      approximate a wider class of functions, including rational functions of low order.
    
    Basis: 1, x, y, z, 1/x, 1/y, 1/z, x*y, x*z, y*z
    """
    c0, c1, c2, c3, c4, c5, c6, c7, c8, c9 = params
    
    # Compute basis elements
    inv_x = 1.0 / x
    inv_y = 1.0 / y
    inv_z = 1.0 / z
    xy = x * y
    xz = x * z
    yz = y * z
    xyz = x * y * z
    inv_xy = 1.0 / (x * y)
    inv_xz = 1.0 / (x * z)
    inv_yz = 1.0 / (y * z)
    inv_xyz = 1.0 / (x * y * z)
    x2 = x * x
    y2 = y * y
    z2 = z * z
    
    # Linear combination with richer basis
    return (c0 + c1 * x + c2 * y + c3 * z + 
            c4 * inv_x + c5 * inv_y + c6 * inv_z + 
            c7 * xy + c8 * xz + c9 * yz)
# EVOLVE-BLOCK-END
