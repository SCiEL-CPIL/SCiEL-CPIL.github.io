"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Structural Change: Hybrid Polynomial-Reciprocal Basis
    # The previous model was a pure quadratic polynomial.
    # This model includes reciprocal terms (1/x, 1/y, 1/z) to capture rational/hyperbolic
    # structures, while retaining linear and pairwise interaction terms.
    # Basis Terms:
    # 0: 1 (Constant)
    # 1: x
    # 2: y
    # 3: z
    # 4: 1/x
    # 5: 1/y
    # 6: 1/z
    # 7: x*y
    # 8: x*z
    # 9: y*z
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Calculate basis terms
    # Inputs are in [0.5, 2], so 1/x is safe and well-conditioned
    inv_x = 1.0 / x
    inv_y = 1.0 / y
    inv_z = 1.0 / z
    
    xy = x * y
    xz = x * z
    yz = y * z
    
    # Linear combination of basis terms
    return p0 + p1 * x + p2 * y + p3 * z + p4 * inv_x + p5 * inv_y + p6 * inv_z + p7 * xy + p8 * xz + p9 * yz
# EVOLVE-BLOCK-END
