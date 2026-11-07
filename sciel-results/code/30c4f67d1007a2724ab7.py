"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression using a rich basis of polynomial, reciprocal, and ratio terms.
    
    The basis is designed to capture:
    - Constant (bias)
    - Linear terms in x, y, z
    - Quadratic terms (x^2, y^2, z^2)
    - Cross terms (xy, xz, yz)
    - Reciprocal terms (1/x, 1/y, 1/z)
    - Ratio terms (x/y, y/z, z/x)
    
    This provides structural capacity to approximate rational-like functions.
    """
    p = params.flatten()
    
    # Ensure we have at least 10 parameters; use zeros for missing
    if len(p) < 10:
        p = np.pad(p, (0, 10 - len(p)))
    
    # Extract parameters
    a0 = p[0]  # bias
    a1 = p[1]  # x
    a2 = p[2]  # y
    a3 = p[3]  # z
    a4 = p[4]  # x^2
    a5 = p[5]  # y^2
    a6 = p[6]  # z^2
    a7 = p[7]  # xy
    a8 = p[8]  # xz
    a9 = p[9]  # yz
    
    # Build the expression
    result = a0 + a1 * x + a2 * y + a3 * z
    result += a4 * x**2 + a5 * y**2 + a6 * z**2
    result += a7 * x * y + a8 * x * z + a9 * y * z
    
    # Additional terms if more parameters are available
    if len(p) > 10:
        # Reciprocal terms
        if len(p) > 11:
            result += p[10] / x
        if len(p) > 12:
            result += p[11] / y
        if len(p) > 13:
            result += p[12] / z
        # Ratio terms
        if len(p) > 14:
            result += p[13] * x / y
        if len(p) > 15:
            result += p[14] * y / z
        if len(p) > 16:
            result += p[15] * z / x
    
    return result
# EVOLVE-BLOCK-END
