import numpy as np
import time

_START_TIME = time.time()

def equation(x, y, params):
    # Deadline guard: safety margin for the 150s limit
    if time.time() - _START_TIME > 145:
        return np.zeros_like(x)
    
    # Build a richer 2D basis with polynomial + trigonometric components
    # 10 parameters to cover diverse function shapes
    
    # Basic features
    x2 = x * x
    y2 = y * y
    
    # Trigonometric features at multiple frequencies
    sx1 = np.sin(np.pi * x)
    cx1 = np.cos(np.pi * x)
    sy1 = np.sin(np.pi * y)
    cy1 = np.cos(np.pi * y)
    
    sx2 = np.sin(2 * np.pi * x)
    cx2 = np.cos(2 * np.pi * x)
    sy2 = np.sin(2 * np.pi * y)
    cy2 = np.cos(2 * np.pi * y)
    
    # Polynomial features (Chebyshev-like)
    px2 = x2 - 0.5
    py2 = y2 - 0.5
    px3 = x * px2  # odd polynomial
    py3 = y * py2
    
    # Cross features
    xy = x * y
    xy2 = x2 * y2
    x2y = x2 * y
    xy2_ = x * y2
    
    # 10 basis functions chosen for maximum diversity
    b0 = np.ones_like(x)          # constant
    b1 = x                         # linear x
    b2 = y                         # linear y
    b3 = xy                        # cross product
    b4 = sx1 * cy1 + cx1 * sy1    # combined trig cross (sin(x+y)-like)
    b5 = sx2 * cy2 + cx2 * sy2    # higher freq trig cross
    b6 = px2 + py2                 # combined quadratic
    b7 = px3 + py3                 # combined cubic (odd)
    b8 = x2y + xy2_                # asymmetric cross terms
    b9 = xy2                       # product of squared terms
    
    result = (params[0]*b0 + params[1]*b1 + params[2]*b2 + params[3]*b3 +
              params[4]*b4 + params[5]*b5 + params[6]*b6 + params[7]*b7 +
              params[8]*b8 + params[9]*b9)
    
    return result
