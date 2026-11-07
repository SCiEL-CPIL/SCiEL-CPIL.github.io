import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Full cubic polynomial basis in 2 variables
    # 10 terms:
    # 1, x, y, x^2, xy, y^2, x^3, x^2y, xy^2, y^3
    
    f1 = np.ones_like(x)
    f2 = x
    f3 = y
    f4 = x**2
    f5 = x * y
    f6 = y**2
    f7 = x**3
    f8 = x**2 * y
    f9 = x * y**2
    f10 = y**3
    
    return (params[0] * f1 + params[1] * f2 + params[2] * f3 + 
            params[3] * f4 + params[4] * f5 + params[5] * f6 + 
            params[6] * f7 + params[7] * f8 + params[8] * f9 + 
            params[9] * f10)
