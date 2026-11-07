import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a Fourier (trigonometric) basis.
    Basis: x, v, sin(x), cos(x), sin(v), cos(v), sin(x+v), cos(x+v), sin(x-v), cos(x-v)
    This explicitly models the periodic phase-space dynamics of the oscillator.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Precompute trigonometric terms
    sin_x = np.sin(x)
    cos_x = np.cos(x)
    sin_v = np.sin(v)
    cos_v = np.cos(v)
    
    sum_xv = x + v
    sin_xv = np.sin(sum_xv)
    cos_xv = np.cos(sum_xv)
    
    diff_xv = x - v
    sin_diff = np.sin(diff_xv)
    cos_diff = np.cos(diff_xv)
    
    return (p0 * x + 
            p1 * v + 
            p2 * sin_x + 
            p3 * cos_x + 
            p4 * sin_v + 
            p5 * cos_v + 
            p6 * sin_xv + 
            p7 * cos_xv + 
            p8 * sin_diff + 
            p9 * cos_diff)
