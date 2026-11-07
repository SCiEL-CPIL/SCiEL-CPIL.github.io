import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression with 2 inputs x, y.
    Uses a parametric trigonometric Fourier series basis, linear in 10 parameters.
    The basis is chosen to capture periodic and smooth non-polynomial structure
    on the domain [-1, 1]^2.
    
    Basis functions:
    0: 1 (bias)
    1: sin(pi*x)
    2: cos(pi*x)
    3: sin(pi*y)
    4: cos(pi*y)
    5: sin(pi*(x+y))
    6: cos(pi*(x-y))
    7: sin(2*pi*x)
    8: sin(2*pi*y)
    9: sin(pi*x)*cos(pi*y)
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Ensure p has at least 10 elements
    if p.size < 10:
        p = np.pad(p, (0, 10 - p.size))
    
    # Compute Fourier basis
    pi = np.pi
    
    b0 = np.ones_like(x)
    
    b1 = np.sin(pi * x)
    b2 = np.cos(pi * x)
    b3 = np.sin(pi * y)
    b4 = np.cos(pi * y)
    
    b5 = np.sin(pi * (x + y))
    b6 = np.cos(pi * (x - y))
    
    b7 = np.sin(2.0 * pi * x)
    b8 = np.sin(2.0 * pi * y)
    
    b9 = np.sin(pi * x) * np.cos(pi * y)
    
    # Combine linearly
    pred = (p[0] * b0 + 
            p[1] * b1 + 
            p[2] * b2 + 
            p[3] * b3 + 
            p[4] * b4 + 
            p[5] * b5 + 
            p[6] * b6 + 
            p[7] * b7 + 
            p[8] * b8 + 
            p[9] * b9)
    
    # Safety: replace non-finite values
    if not np.all(np.isfinite(pred)):
        pred = np.where(np.isfinite(pred), pred, 0.0)
    
    return pred
