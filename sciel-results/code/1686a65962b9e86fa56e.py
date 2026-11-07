import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression using a separable Chebyshev basis:
    f(x,y) = sum_{i=0}^{4} a_i T_i(x) + sum_{j=0}^{4} b_j T_j(y)
    
    This captures additive structures in x and y independently, which is
    well-conditioned for BFGS optimization.
    
    Chebyshev polynomials of the first kind:
    T_0(t) = 1
    T_1(t) = t
    T_2(t) = 2t^2 - 1
    T_3(t) = 4t^3 - 3t
    T_4(t) = 8t^4 - 8t^2 + 1
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    
    p = params
    
    # Compute Chebyshev polynomials for x (degrees 0-4)
    T0_x = np.ones_like(x)
    T1_x = x
    T2_x = 2.0 * x * x - 1.0
    T3_x = 4.0 * x * x * x - 3.0 * x
    T4_x = 8.0 * x**4 - 8.0 * x * x + 1.0
    
    # Compute Chebyshev polynomials for y (degrees 0-4)
    T0_y = np.ones_like(y)
    T1_y = y
    T2_y = 2.0 * y * y - 1.0
    T3_y = 4.0 * y * y * y - 3.0 * y
    T4_y = 8.0 * y**4 - 8.0 * y * y + 1.0
    
    # Separable sum: 5 terms in x + 5 terms in y
    pred = (p[0] * T0_x + p[1] * T1_x + p[2] * T2_x + p[3] * T3_x + p[4] * T4_x +
            p[5] * T0_y + p[6] * T1_y + p[7] * T2_y + p[8] * T3_y + p[9] * T4_y)
    
    # Safety check for finite values
    if not np.all(np.isfinite(pred)):
        pred = np.where(np.isfinite(pred), pred, 0.0)
    
    return pred
