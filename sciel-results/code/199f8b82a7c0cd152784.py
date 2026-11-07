import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximates the target function using a hybrid polynomial-trigonometric basis.
    
    Basis functions:
    0: 1
    1: x
    2: y
    3: x^2
    4: y^2
    5: x*y
    6: sin(x)
    7: sin(y)
    8: sin(x+y)
    9: sin(x-y)
    
    This basis covers linear, quadratic, and periodic behaviors, which are common
    in symbolic regression benchmarks.
    """
    # Polynomial terms
    x2 = x * x
    y2 = y * y
    xy = x * y

    # Trigonometric terms
    sin_x = np.sin(x)
    sin_y = np.sin(y)
    sin_xy_plus = np.sin(x + y)
    sin_xy_minus = np.sin(x - y)

    # Construct prediction
    pred = (params[0] 
            + params[1] * x 
            + params[2] * y 
            + params[3] * x2 
            + params[4] * y2 
            + params[5] * xy 
            + params[6] * sin_x 
            + params[7] * sin_y 
            + params[8] * sin_xy_plus 
            + params[9] * sin_xy_minus)
            
    return pred
