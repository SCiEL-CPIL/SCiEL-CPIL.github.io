"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a separable tensor-product Chebyshev basis.
    
    We use Chebyshev polynomials T0, T1, T2 for each variable.
    The basis is the full tensor product: T_i(x) * T_j(y) for i,j in {0,1,2}.
    This gives 9 terms, but we have 10 params. We add one more term:
    T_3(x) * T_0(y) to capture some cubic behavior in x.
    
    Chebyshev polynomials:
    T_0(t) = 1
    T_1(t) = t
    T_2(t) = 2t^2 - 1
    T_3(t) = 4t^3 - 3t
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Compute Chebyshev polynomials for x and y
    T0_x = np.ones_like(x)
    T1_x = x
    T2_x = 2.0 * x**2 - 1.0
    T3_x = 4.0 * x**3 - 3.0 * x
    
    T0_y = np.ones_like(y)
    T1_y = y
    T2_y = 2.0 * y**2 - 1.0
    
    # Tensor product basis terms
    # (i,j) pairs for i in {0,1,2}, j in {0,1,2} plus (3,0)
    # 0: T0(x)*T0(y) = 1
    # 1: T1(x)*T0(y) = x
    # 2: T0(x)*T1(y) = y
    # 3: T2(x)*T0(y) = 2x^2 - 1
    # 4: T1(x)*T1(y) = x*y
    # 5: T0(x)*T2(y) = 2y^2 - 1
    # 6: T2(x)*T1(y) = (2x^2-1)*y
    # 7: T1(x)*T2(y) = x*(2y^2-1)
    # 8: T2(x)*T2(y) = (2x^2-1)*(2y^2-1)
    # 9: T3(x)*T0(y) = 4x^3 - 3x
    
    return (
        p0 * (T0_x * T0_y)
        + p1 * (T1_x * T0_y)
        + p2 * (T0_x * T1_y)
        + p3 * (T2_x * T0_y)
        + p4 * (T1_x * T1_y)
        + p5 * (T0_x * T2_y)
        + p6 * (T2_x * T1_y)
        + p7 * (T1_x * T2_y)
        + p8 * (T2_x * T2_y)
        + p9 * (T3_x * T0_y)
    )
# EVOLVE-BLOCK-END
