import numpy as np
from scipy.optimize import minimize
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    
    # Build a richer basis using polynomial features up to degree 4
    # and interaction terms, plus some trigonometric features for periodicity
    n = x.shape[0]
    
    # Polynomial features for x
    x1 = x
    x2 = x ** 2
    x3 = x ** 3
    x4 = x ** 4
    
    # Polynomial features for y
    y1 = y
    y2 = y ** 2
    y3 = y ** 3
    y4 = y ** 4
    
    # Interaction terms
    xy = x1 * y1
    x2y = x2 * y1
    xy2 = x1 * y2
    x2y2 = x2 * y2
    
    # Trig features (in case the function has periodic components)
    sin_x = np.sin(np.pi * x)
    cos_x = np.cos(np.pi * x)
    sin_y = np.sin(np.pi * y)
    cos_y = np.cos(np.pi * y)
    sin_xy = np.sin(np.pi * x * y)
    cos_xy = np.cos(np.pi * x * y)
    
    # Radial feature
    r2 = x ** 2 + y ** 2
    r = np.sqrt(r2 + 1e-10)
    exp_r = np.exp(-r)
    
    # Basis: 10 features selected to span a good function space
    # 1. Constant
    b1 = np.ones_like(x)
    
    # 2. x
    b2 = x1
    
    # 3. y
    b3 = y1
    
    # 4. x^2
    b4 = x2
    
    # 5. y^2
    b5 = y2
    
    # 6. xy
    b6 = xy
    
    # 7. x^2 * y
    b7 = x2y
    
    # 8. x * y^2
    b8 = xy2
    
    # 9. x^2 * y^2
    b9 = x2y2
    
    # 10. exp(-sqrt(x^2+y^2))
    b10 = exp_r
    
    return (params[0] * b1 +
            params[1] * b2 +
            params[2] * b3 +
            params[3] * b4 +
            params[4] * b5 +
            params[5] * b6 +
            params[6] * b7 +
            params[7] * b8 +
            params[8] * b9 +
            params[9] * b10)
