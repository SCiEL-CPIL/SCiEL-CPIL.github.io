"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Chebyshev polynomial basis for better conditioning and approximation.
    Basis terms:
    0: 1
    1: T_1(x) = x
    2: T_2(x) = 2x^2 - 1
    3: T_1(y) = y
    4: T_2(y) = 2y^2 - 1
    5: T_1(x)T_1(y) = xy
    6: T_2(x)T_1(y) = (2x^2 - 1)y
    7: T_1(x)T_2(y) = x(2y^2 - 1)
    8: T_2(x)T_2(y) = (2x^2 - 1)(2y^2 - 1)
    9: T_3(x) = 4x^3 - 3x
    """
    p = params
    x = np.asarray(x, dtype=np.float64).flatten()
    y = np.asarray(y, dtype=np.float64).flatten()
    
    # Chebyshev polynomials
    T1_x = x
    T2_x = 2 * x * x - 1
    T3_x = 4 * x * x * x - 3 * x
    
    T1_y = y
    T2_y = 2 * y * y - 1
    
    pred = (
        p[0] +
        p[1] * T1_x +
        p[2] * T2_x +
        p[3] * T1_y +
        p[4] * T2_y +
        p[5] * T1_x * T1_y +
        p[6] * T2_x * T1_y +
        p[7] * T1_x * T2_y +
        p[8] * T2_x * T2_y +
        p[9] * T3_x
    )
    return pred
# EVOLVE-BLOCK-END
