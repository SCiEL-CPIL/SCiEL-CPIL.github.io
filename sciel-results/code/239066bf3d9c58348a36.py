import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression for a nonlinear oscillator using a trigonometric basis.
    
    The basis functions are chosen to capture periodic behavior and interactions
    between position (x) and velocity (v), which is more physically plausible
    for an oscillator than a pure polynomial expansion.
    
    Basis:
    1 (bias)
    sin(x)
    cos(x)
    sin(v)
    cos(v)
    sin(x + v)
    cos(x + v)
    sin(x - v)
    cos(x - v)
    x * sin(v)  (interaction term)
    """
    p = params
    
    # Precompute trigonometric functions
    sin_x = np.sin(x)
    cos_x = np.cos(x)
    sin_v = np.sin(v)
    cos_v = np.cos(v)
    
    # Combined phase arguments
    sin_xv = np.sin(x + v)
    cos_xv = np.cos(x + v)
    sin_x_minus_v = np.sin(x - v)
    cos_x_minus_v = np.cos(x - v)
    
    # Interaction term
    x_sin_v = x * sin_v
    
    # Combine with parameters
    out = (p[0] * 1.0 +
           p[1] * sin_x +
           p[2] * cos_x +
           p[3] * sin_v +
           p[4] * cos_v +
           p[5] * sin_xv +
           p[6] * cos_xv +
           p[7] * sin_x_minus_v +
           p[8] * cos_x_minus_v +
           p[9] * x_sin_v)
           
    return out
