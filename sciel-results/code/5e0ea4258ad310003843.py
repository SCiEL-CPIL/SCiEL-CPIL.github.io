import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Equation for symbolic regression with 2 inputs x, y.
    
    Uses a combined basis of Chebyshev polynomials (for stable polynomial approximation)
    and Gaussian RBF terms (for local non-polynomial structure). The model is linear
    in the 10 parameters, which allows BFGS to converge to the global minimum.
    
    Basis functions:
    1. 1 (bias)
    2. T1(x) = x
    3. T1(y) = y
    4. T2(x) = 2x^2 - 1
    5. T2(y) = 2y^2 - 1
    6. T1(x)*T1(y) = x*y
    7. RBF centered at (0,0)
    8. RBF centered at (0.5,0.5)
    9. RBF centered at (-0.5,-0.5)
    10. RBF centered at (0.5,-0.5)
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Ensure 1D arrays
    x = x.ravel()
    y = y.ravel()
    
    # Chebyshev polynomial basis
    b1 = np.ones_like(x)           # T0 = 1
    b2 = x                         # T1(x) = x
    b3 = y                         # T1(y) = y
    b4 = 2.0 * x * x - 1.0         # T2(x) = 2x^2 - 1
    b5 = 2.0 * y * y - 1.0         # T2(y) = 2y^2 - 1
    b6 = x * y                     # T1(x)*T1(y) = x*y
    
    # Gaussian RBF terms with different centers and scales
    r1_sq = x * x + y * y
    b7 = np.exp(-4.0 * r1_sq)
    
    r2_sq = (x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5)
    b8 = np.exp(-3.0 * r2_sq)
    
    r3_sq = (x + 0.5) * (x + 0.5) + (y + 0.5) * (y + 0.5)
    b9 = np.exp(-3.0 * r3_sq)
    
    r4_sq = (x - 0.5) * (x - 0.5) + (y + 0.5) * (y + 0.5)
    b10 = np.exp(-3.0 * r4_sq)
    
    # Linear combination
    pred = (p[0] * b1 +
            p[1] * b2 +
            p[2] * b3 +
            p[3] * b4 +
            p[4] * b5 +
            p[5] * b6 +
            p[6] * b7 +
            p[7] * b8 +
            p[8] * b9 +
            p[9] * b10)
    
    # Safety check for non-finite values
    if not np.all(np.isfinite(pred)):
        pred = np.where(np.isfinite(pred), pred, 0.0)
    
    return pred
