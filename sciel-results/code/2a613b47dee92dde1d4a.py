import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a Fourier series basis.
    
    The model is linear in 10 parameters, which are fitted by the external BFGS optimizer.
    The basis functions are chosen to be a set of low-frequency sine and cosine functions
    that form a reasonably complete set for smooth functions on [-1, 1] x [-1, 1].
    
    Basis functions (10 terms):
    0: 1                          (bias)
    1: sin(pi*x)                  (fundamental x)
    2: cos(pi*x)                  (fundamental x)
    3: sin(pi*y)                  (fundamental y)
    4: cos(pi*y)                  (fundamental y)
    5: sin(pi*(x+y))              (diagonal interaction)
    6: cos(pi*(x+y))              (diagonal interaction)
    7: sin(pi*(x-y))              (anti-diagonal interaction)
    8: cos(pi*(x-y))              (anti-diagonal interaction)
    9: sin(2*pi*x)*cos(2*pi*y)    (higher-order interaction)
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    p = np.asarray(params, dtype=np.float64).ravel()
    
    if p.size < 10:
        p = np.pad(p, (0, 10 - p.size))
    
    # Fourier basis
    b1 = np.ones_like(x)
    b2 = np.sin(np.pi * x)
    b3 = np.cos(np.pi * x)
    b4 = np.sin(np.pi * y)
    b5 = np.cos(np.pi * y)
    
    xy_sum = x + y
    xy_diff = x - y
    
    b6 = np.sin(np.pi * xy_sum)
    b7 = np.cos(np.pi * xy_sum)
    b8 = np.sin(np.pi * xy_diff)
    b9 = np.cos(np.pi * xy_diff)
    
    b10 = np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)
    
    # Linear combination
    out = (p[0] * b1 + 
           p[1] * b2 + 
           p[2] * b3 + 
           p[3] * b4 + 
           p[4] * b5 + 
           p[5] * b6 + 
           p[6] * b7 + 
           p[7] * b8 + 
           p[8] * b9 + 
           p[9] * b10)
    
    # Safety check
    if not np.all(np.isfinite(out)):
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        
    return out
