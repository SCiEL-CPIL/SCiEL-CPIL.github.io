import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Hybrid basis: combine Chebyshev polynomials (good for smooth functions)
    # with trigonometric terms (good for periodic/oscillatory behavior)
    # and polynomial interactions to capture non-linear couplings.
    
    # Polynomial terms in x
    x2 = x * x
    x3 = x2 * x
    
    # Polynomial terms in y
    y2 = y * y
    y3 = y2 * y
    
    # Interaction terms
    xy = x * y
    x2y = x2 * y
    y2x = y2 * x
    
    # Trigonometric terms
    sin_px = np.sin(np.pi * x)
    cos_px = np.cos(np.pi * x)
    sin_py = np.sin(np.pi * y)
    cos_py = np.cos(np.pi * y)
    
    # Basis selection:
    # 0: 1 (constant)
    # 1: x (linear x)
    # 2: y (linear y)
    # 3: x^2 (quadratic x)
    # 4: y^2 (quadratic y)
    # 5: xy (interaction)
    # 6: sin(pi*x) (trig x)
    # 7: sin(pi*y) (trig y)
    # 8: cos(pi*x) (trig x)
    # 9: cos(pi*y) (trig y)
    
    f = (params[0] * 1.0 +
         params[1] * x +
         params[2] * y +
         params[3] * x2 +
         params[4] * y2 +
         params[5] * xy +
         params[6] * sin_px +
         params[7] * sin_py +
         params[8] * cos_px +
         params[9] * cos_py)
         
    return f
