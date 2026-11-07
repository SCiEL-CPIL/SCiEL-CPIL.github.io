import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a separable tensor-product Chebyshev basis.
    
    Basis: T_i(x) * T_j(y) for i, j in {0, 1, 2}.
    This spans polynomials of degree <= 2 in x and <= 2 in y.
    """
    # Compute Chebyshev polynomials for x and y
    T0_x = np.ones_like(x)
    T1_x = x
    T2_x = 2 * x**2 - 1
    
    T0_y = np.ones_like(y)
    T1_y = y
    T2_y = 2 * y**2 - 1
    
    # Construct the 9 basis functions (3x3 grid)
    # Row i, Col j -> T_i(x) * T_j(y)
    b0 = T0_x * T0_y  # 1
    b1 = T1_x * T0_y  # x
    b2 = T2_x * T0_y  # 2x^2 - 1
    
    b3 = T0_x * T1_y  # y
    b4 = T1_x * T1_y  # xy
    b5 = T2_x * T1_y  # (2x^2 - 1)y
    
    b6 = T0_x * T2_y  # 2y^2 - 1
    b7 = T1_x * T2_y  # x(2y^2 - 1)
    b8 = T2_x * T2_y  # (2x^2 - 1)(2y^2 - 1)
    
    # Linear combination using first 9 params
    f = (params[0] * b0 +
         params[1] * b1 +
         params[2] * b2 +
         params[3] * b3 +
         params[4] * b4 +
         params[5] * b5 +
         params[6] * b6 +
         params[7] * b7 +
         params[8] * b8)
    
    return f
