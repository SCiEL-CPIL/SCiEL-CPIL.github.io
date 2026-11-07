"""Symbolic regression: 10-parameter Chebyshev basis with cross terms for 2D function recovery."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    10-parameter Chebyshev polynomial basis on [-1,1]^2 with rich interaction terms.
    
    Features:
      0: T0(x)*T0(y) = 1           (constant)
      1: T1(x)                    (linear x)
      2: T1(y)                    (linear y)
      3: T2(x)                    (quadratic x)
      4: T2(y)                    (quadratic y)
      5: T1(x)*T1(y)              (bilinear interaction)
      6: T2(x)*T1(y)              (quadratic-x, linear-y)
      7: T1(x)*T2(y)              (linear-x, quadratic-y)
      8: T2(x)*T2(y)              (quadratic-quadratic)
      9: T3(x)*T3(y)              (cubic-cubic interaction)
    """
    p = np.asarray(params, dtype=float)
    
    # Chebyshev polynomials for x
    T0x = np.ones_like(x, dtype=float)
    T1x = x
    T2x = 2.0 * x * x - 1.0
    T3x = 4.0 * x**3 - 3.0 * x
    
    # Chebyshev polynomials for y
    T0y = np.ones_like(y, dtype=float)
    T1y = y
    T2y = 2.0 * y * y - 1.0
    T3y = 4.0 * y**3 - 3.0 * y
    
    pred = (
        p[0] * T0x * T0y +
        p[1] * T1x +
        p[2] * T1y +
        p[3] * T2x +
        p[4] * T2y +
        p[5] * T1x * T1y +
        p[6] * T2x * T1y +
        p[7] * T1x * T2y +
        p[8] * T2x * T2y +
        p[9] * T3x * T3y
    )
    
    return pred
# EVOLVE-BLOCK-END
