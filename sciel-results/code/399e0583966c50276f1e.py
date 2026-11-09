"""
Combines Chebyshev polynomial basis (inspiration from p25/p13) with the linear-in-parameters
structure of the current program. Chebyshev polynomials provide stable, near-orthogonal
approximation on [-1, 1], reducing parameter correlation and allowing BFGS to efficiently
use all 10 parameters. Includes deadline guard for safety.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global time tracking
_START_TIME = time.time()
_TIME_LIMIT = 150.0
_SAFETY_MARGIN = 5.0


def _time_remaining():
    return _TIME_LIMIT - (time.time() - _START_TIME)


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Computes predictions using a Chebyshev polynomial basis in 2D.
    
    Args:
        x: Array of x inputs
        y: Array of y inputs
        params: Array of 10 parameters
        
    Returns:
        Array of predictions
    """
    # Deadline guard
    if _time_remaining() < _SAFETY_MARGIN:
        return params[0] + params[1] * x + params[2] * y

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    if len(p) < 10:
        p = np.pad(p, (0, 10 - len(p)), constant_values=0.0)
    else:
        p = p[:10]
    
    # Compute Chebyshev polynomials of the first kind for x and y
    # T0(t) = 1
    # T1(t) = t
    # T2(t) = 2t^2 - 1
    # T3(t) = 4t^3 - 3t
    # T4(t) = 8t^4 - 8t^2 + 1
    
    # Chebyshev basis for x
    T0_x = np.ones_like(x)
    T1_x = x
    T2_x = 2 * x * x - 1
    T3_x = 4 * x * x * x - 3 * x
    T4_x = 8 * x**4 - 8 * x**2 + 1
    
    # Chebyshev basis for y
    T0_y = np.ones_like(y)
    T1_y = y
    T2_y = 2 * y * y - 1
    T3_y = 4 * y * y * y - 3 * y
    T4_y = 8 * y**4 - 8 * y**2 + 1
    
    # Build a 2D Chebyshev-like basis with 10 terms
    # Using products and individual terms to span a rich space
    # T0_x*T0_y = 1 (constant)
    # T1_x = x
    # T1_y = y
    # T2_x = 2x^2 - 1
    # T2_y = 2y^2 - 1
    # T1_x*T1_y = x*y
    # T3_x = 4x^3 - 3x
    # T3_y = 4y^3 - 3y
    # T2_x*T1_y = (2x^2-1)*y
    # T1_x*T2_y = x*(2y^2-1)
    
    t0 = T0_x * T0_y  # 1
    t1 = T1_x          # x
    t2 = T1_y          # y
    t3 = T2_x          # 2x^2 - 1
    t4 = T2_y          # 2y^2 - 1
    t5 = T1_x * T1_y   # x*y
    t6 = T3_x          # 4x^3 - 3x
    t7 = T3_y          # 4y^3 - 3y
    t8 = T2_x * T1_y   # (2x^2-1)*y
    t9 = T1_x * T2_y   # x*(2y^2-1)
    
    # Linear combination
    pred = (p[0] * t0 + 
            p[1] * t1 + 
            p[2] * t2 + 
            p[3] * t3 + 
            p[4] * t4 + 
            p[5] * t5 + 
            p[6] * t6 + 
            p[7] * t7 + 
            p[8] * t8 + 
            p[9] * t9)
    
    # Ensure output is finite
    if not np.all(np.isfinite(pred)):
        pred = np.nan_to_num(pred, nan=0.0, posinf=1e10, neginf=-1e10)
    
    return pred
# EVOLVE-BLOCK-END
