"""Symbolic regression: 10-parameter Chebyshev basis with cross terms for 2D function recovery."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _chebyshev(n, t):
    """Compute Chebyshev polynomial T_n(t) for array t."""
    if n == 0:
        return np.ones_like(t, dtype=float)
    elif n == 1:
        return t.copy()
    else:
        T_nm2 = np.ones_like(t, dtype=float)
        T_nm1 = t.copy()
        for k in range(2, n + 1):
            T_n = 2.0 * t * T_nm1 - T_nm2
            T_nm2 = T_nm1
            T_nm1 = T_n
        return T_nm1


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    10-parameter Chebyshev basis with cross terms for 2D function recovery.
    
    Basis functions:
    T0(x), T1(x), T2(x), T0(y), T1(y), T2(y), T1(x)*T1(y), T2(x)*T1(y), T1(x)*T2(y), T2(x)*T2(y)
    
    This spans constant, linear, quadratic terms in x and y, plus interaction terms,
    providing good coverage of smooth functions on [-1,1]^2.
    """
    # Compute Chebyshev polynomials up to degree 2
    T0x = np.ones_like(x, dtype=float)
    T1x = x.copy()
    T2x = 2.0 * x * x - 1.0
    
    T0y = np.ones_like(y, dtype=float)
    T1y = y.copy()
    T2y = 2.0 * y * y - 1.0
    
    # Build 10 basis functions
    f0 = T0x                          # constant
    f1 = T1x                          # linear x
    f2 = T2x                          # quadratic x
    f3 = T0y                          # constant in y (redundant with f0, but BFGS can handle)
    f4 = T1y                          # linear y
    f5 = T2y                          # quadratic y
    f6 = T1x * T1y                    # interaction xy
    f7 = T2x * T1y                    # x^2 * y
    f8 = T1x * T2y                    # x * y^2
    f9 = T2x * T2y                    # x^2 * y^2
    
    return (params[0] * f0 + params[1] * f1 + params[2] * f2 +
            params[3] * f3 + params[4] * f4 + params[5] * f5 +
            params[6] * f6 + params[7] * f7 + params[8] * f8 +
            params[9] * f9)
# EVOLVE-BLOCK-END
