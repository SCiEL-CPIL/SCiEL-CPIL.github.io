import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator model using a trigonometric basis.
    This approach uses sine and cosine functions of position and velocity,
    along with mixed terms, to capture nonlinear periodic or bounded dynamics.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Precompute trigonometric functions
    sin_x = np.sin(x)
    cos_x = np.cos(x)
    sin_v = np.sin(v)
    cos_v = np.cos(v)
    
    # Construct the equation
    # p0: sin(x)
    # p1: cos(x)
    # p2: sin(v)
    # p3: cos(v)
    # p4: x * sin(v)
    # p5: v * cos(x)
    # p6: sin(x) * v
    # p7: x * cos(v)
    # p8: cos(x) * cos(v)
    # p9: constant
    
    return (p0 * sin_x + 
            p1 * cos_x + 
            p2 * sin_v + 
            p3 * cos_v + 
            p4 * x * sin_v + 
            p5 * v * cos_x + 
            p6 * sin_x * v + 
            p7 * x * cos_v + 
            p8 * cos_x * cos_v + 
            p9)
