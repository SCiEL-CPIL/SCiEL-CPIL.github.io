import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    
    # Complete bivariate cubic polynomial basis (10 terms)
    # 1, x, y, x^2, xy, y^2, x^3, x^2*y, x*y^2, y^3
    b0 = np.ones_like(x)
    b1 = x
    b2 = y
    b3 = x**2
    b4 = x * y
    b5 = y**2
    b6 = x**3
    b7 = x**2 * y
    b8 = x * y**2
    b9 = y**3
    
    return (params[0] * b0 +
            params[1] * b1 +
            params[2] * b2 +
            params[3] * b3 +
            params[4] * b4 +
            params[5] * b5 +
            params[6] * b6 +
            params[7] * b7 +
            params[8] * b8 +
            params[9] * b9)
