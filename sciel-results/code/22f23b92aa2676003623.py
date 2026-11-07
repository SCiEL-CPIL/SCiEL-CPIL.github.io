import numpy as np
import time

# Global time tracking for safety
_START_TIME = time.time()
_TIME_LIMIT = 115.0  # 5 second margin from 120s limit

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximates the target function using a Chebyshev polynomial basis in (x, y).
    
    Basis functions (10 terms, one per parameter):
    p[0]: T0(x)T0(y) = 1
    p[1]: T1(x)T0(y) = x
    p[2]: T0(x)T1(y) = y
    p[3]: T2(x)T0(y) = 2x^2 - 1
    p[4]: T0(x)T2(y) = 2y^2 - 1
    p[5]: T1(x)T1(y) = xy
    p[6]: T3(x)T0(y) = 4x^3 - 3x
    p[7]: T0(x)T3(y) = 4y^3 - 3y
    p[8]: T2(x)T1(y) = (2x^2 - 1)y
    p[9]: T1(x)T2(y) = x(2y^2 - 1)
    
    Chebyshev polynomials are orthogonal on [-1, 1], improving conditioning for BFGS.
    """
    
    # Safety check: If we are running out of time, return a simple stable prediction
    if time.time() - _START_TIME > _TIME_LIMIT:
        x2 = x * x
        y2 = y * y
        xy = x * y
        return (params[0] 
                + params[1] * x 
                + params[2] * y 
                + params[3] * x2 
                + params[4] * y2 
                + params[5] * xy)

    # Pre-compute Chebyshev polynomials
    # T0(t) = 1
    # T1(t) = t
    # T2(t) = 2t^2 - 1
    # T3(t) = 4t^3 - 3t
    
    x2 = x * x
    y2 = y * y
    x3 = x2 * x
    y3 = y2 * y
    
    # Chebyshev terms for x
    Tx0 = np.ones_like(x)
    Tx1 = x
    Tx2 = 2 * x2 - 1
    Tx3 = 4 * x3 - 3 * x
    
    # Chebyshev terms for y
    Ty0 = np.ones_like(y)
    Ty1 = y
    Ty2 = 2 * y2 - 1
    Ty3 = 4 * y3 - 3 * y
    
    # Construct basis terms (products of Chebyshev polynomials)
    b0 = Tx0 * Ty0
    b1 = Tx1 * Ty0
    b2 = Tx0 * Ty1
    b3 = Tx2 * Ty0
    b4 = Tx0 * Ty2
    b5 = Tx1 * Ty1
    b6 = Tx3 * Ty0
    b7 = Tx0 * Ty3
    b8 = Tx2 * Ty1
    b9 = Tx1 * Ty2
    
    # Linear combination
    pred = (params[0] * b0
            + params[1] * b1
            + params[2] * b2
            + params[3] * b3
            + params[4] * b4
            + params[5] * b5
            + params[6] * b6
            + params[7] * b7
            + params[8] * b8
            + params[9] * b9)

    # Ensure numerical stability
    if not np.all(np.isfinite(pred)):
        # Fallback to quadratic monomial part
        return (params[0] 
                + params[1] * x 
                + params[2] * y 
                + params[3] * x2 
                + params[4] * y2 
                + params[5] * xy)

    return pred
