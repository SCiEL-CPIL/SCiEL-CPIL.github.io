"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Computes the prediction using a 2D Chebyshev polynomial basis of total degree <= 3.
    This basis provides better conditioning for BFGS than monomials on [-1, 1].
    
    Basis functions:
    0: T0(x)T0(y) = 1
    1: T1(x)T0(y) = x
    2: T0(x)T1(y) = y
    3: T2(x)T0(y) = 2x^2 - 1
    4: T1(x)T1(y) = xy
    5: T0(x)T2(y) = 2y^2 - 1
    6: T3(x)T0(y) = 4x^3 - 3x
    7: T2(x)T1(y) = (2x^2 - 1)y
    8: T1(x)T2(y) = x(2y^2 - 1)
    9: T0(x)T3(y) = 4y^3 - 3y
    """
    # Compute Chebyshev polynomials for x and y
    T0_x = np.ones_like(x)
    T1_x = x
    T2_x = 2 * x**2 - 1
    T3_x = 4 * x**3 - 3 * x

    T0_y = np.ones_like(y)
    T1_y = y
    T2_y = 2 * y**2 - 1
    T3_y = 4 * y**3 - 3 * y

    # Construct the 10 basis terms
    # Order matters for consistency, but any fixed order works as long as params align
    b0 = T0_x * T0_y  # 1
    b1 = T1_x * T0_y  # x
    b2 = T0_x * T1_y  # y
    b3 = T2_x * T0_y  # 2x^2 - 1
    b4 = T1_x * T1_y  # xy
    b5 = T0_x * T2_y  # 2y^2 - 1
    b6 = T3_x * T0_y  # 4x^3 - 3x
    b7 = T2_x * T1_y  # (2x^2 - 1)y
    b8 = T1_x * T2_y  # x(2y^2 - 1)
    b9 = T0_x * T3_y  # 4y^3 - 3y

    # Combine with parameters
    # params is an array of length 10
    pred = (params[0] * b0 + 
            params[1] * b1 + 
            params[2] * b2 + 
            params[3] * b3 + 
            params[4] * b4 + 
            params[5] * b5 + 
            params[6] * b6 + 
            params[7] * b7 + 
            params[8] * b8 + 
            params[9] * b9)
            
    return pred
# EVOLVE-BLOCK-END
