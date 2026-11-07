import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximates the target function using a mixed polynomial and trigonometric basis.
    f(x, y) = p0 + p1*x + p2*y + p3*x^2 + p4*y^2 + p5*x*y 
             + p6*sin(x) + p7*sin(y) + p8*cos(x) + p9*cos(y)
    
    This combines polynomial terms with trigonometric functions to capture
    both polynomial and periodic behaviors in the hidden function.
    """
    # Polynomial terms
    x2 = x * x
    y2 = y * y
    xy = x * y

    # Trigonometric terms
    sinx = np.sin(x)
    siny = np.sin(y)
    cosx = np.cos(x)
    cosy = np.cos(y)

    # Construct prediction
    # Order: 1, x, y, x^2, y^2, xy, sin(x), sin(y), cos(x), cos(y)
    pred = (params[0] 
            + params[1] * x 
            + params[2] * y 
            + params[3] * x2 
            + params[4] * y2 
            + params[5] * xy 
            + params[6] * sinx 
            + params[7] * siny 
            + params[8] * cosx 
            + params[9] * cosy)
            
    return pred
