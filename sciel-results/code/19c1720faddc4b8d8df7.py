"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression using a separable Chebyshev tensor-product basis.
    
    Uses 10 basis functions:
      T_0(x)T_0(y), T_1(x)T_0(y), T_0(x)T_1(y), T_1(x)T_1(y),
      T_2(x)T_0(y), T_0(x)T_2(y), T_2(x)T_1(y), T_1(x)T_2(y), T_2(x)T_2(y),
      T_3(x)T_1(y)
    
    Chebyshev polynomials on [-1,1]:
      T_0(t) = 1
      T_1(t) = t
      T_2(t) = 2t^2 - 1
      T_3(t) = 4t^3 - 3t
    """
    # Chebyshev polynomials for x and y
    T0x = np.ones_like(x)
    T1x = x
    T2x = 2.0 * x**2 - 1.0
    T3x = 4.0 * x**3 - 3.0 * x

    T0y = np.ones_like(y)
    T1y = y
    T2y = 2.0 * y**2 - 1.0

    # 10 basis functions (tensor products)
    basis = np.column_stack([
        T0x * T0y,   # 1
        T1x * T0y,   # x
        T0x * T1y,   # y
        T1x * T1y,   # x*y
        T2x * T0y,   # 2x^2 - 1
        T0x * T2y,   # 2y^2 - 1
        T2x * T1y,   # (2x^2-1)*y
        T1x * T2y,   # x*(2y^2-1)
        T2x * T2y,   # (2x^2-1)*(2y^2-1)
        T3x * T1y,   # (4x^3-3x)*y
    ])

    return basis @ params
# EVOLVE-BLOCK-END
