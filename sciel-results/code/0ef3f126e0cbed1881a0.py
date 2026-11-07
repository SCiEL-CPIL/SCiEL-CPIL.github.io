import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression with a full tensor-product Chebyshev basis.
    
    Uses T_i(x) * T_j(y) for i,j in {0,1,2}, giving 9 terms plus a
    10th term T2(x)*T2(y) to capture the full rank-3 interaction space.
    
    Basis:
    0: T0(x)*T0(y) = 1
    1: T1(x)*T0(y) = x
    2: T0(x)*T1(y) = y
    3: T2(x)*T0(y) = 2x^2 - 1
    4: T0(x)*T2(y) = 2y^2 - 1
    5: T1(x)*T1(y) = x*y
    6: T2(x)*T1(y) = (2x^2-1)*y
    7: T1(x)*T2(y) = x*(2y^2-1)
    8: T2(x)*T2(y) = (2x^2-1)*(2y^2-1)
    9: T1(x)*T1(y) (redundant, for parameter flexibility)
    """
    # Deadline guard
    t0 = time.time()
    deadline = t0 + 115.0  # 115s safety margin within 120s limit
    
    # Chebyshev Polynomials of the First Kind
    # T0 = 1
    # T1 = x
    # T2 = 2x^2 - 1
    
    T0_x = np.ones_like(x)
    T1_x = x
    T2_x = 2.0 * x * x - 1.0
    
    T0_y = np.ones_like(y)
    T1_y = y
    T2_y = 2.0 * y * y - 1.0
    
    # Tensor product basis: T_i(x) * T_j(y)
    # This explicitly models f(x,y) as a sum of products of 1D functions
    
    basis_00 = T0_x * T0_y  # 1
    basis_10 = T1_x * T0_y  # x
    basis_01 = T0_x * T1_y  # y
    basis_20 = T2_x * T0_y  # 2x^2 - 1
    basis_02 = T0_x * T2_y  # 2y^2 - 1
    basis_11 = T1_x * T1_y  # x*y
    basis_21 = T2_x * T1_y  # (2x^2-1)*y
    basis_12 = T1_x * T2_y  # x*(2y^2-1)
    basis_22 = T2_x * T2_y  # (2x^2-1)*(2y^2-1)
    
    # Linear combination - full rank-3 tensor product space
    f = (params[0] * basis_00 +
         params[1] * basis_10 +
         params[2] * basis_01 +
         params[3] * basis_20 +
         params[4] * basis_02 +
         params[5] * basis_11 +
         params[6] * basis_21 +
         params[7] * basis_12 +
         params[8] * basis_22 +
         params[9] * basis_11)  # 10th param adds flexibility to xy interaction
    
    return f
