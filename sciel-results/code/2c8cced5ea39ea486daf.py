import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a hybrid basis: low-degree polynomials + Gaussian RBFs.
    
    The model is linear in 10 parameters, which are fitted by the external BFGS optimizer.
    
    Basis functions (10 terms):
    0: 1                          (bias)
    1: x                          (linear x)
    2: y                          (linear y)
    3: x*y                        (interaction)
    4: x^2                        (quadratic x)
    5: y^2                        (quadratic y)
    6: exp(-2*((x-0.5)^2 + (y-0.5)^2))   (RBF center (0.5, 0.5))
    7: exp(-2*((x+0.5)^2 + (y-0.5)^2))   (RBF center (-0.5, 0.5))
    8: exp(-2*((x-0.5)^2 + (y+0.5)^2))   (RBF center (0.5, -0.5))
    9: exp(-2*((x+0.5)^2 + (y+0.5)^2))   (RBF center (-0.5, -0.5))
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    p = np.asarray(params, dtype=np.float64).ravel()
    
    if p.size < 10:
        p = np.pad(p, (0, 10 - p.size))
    
    # Polynomial basis
    b1 = np.ones_like(x)
    b2 = x
    b3 = y
    b4 = x * y
    b5 = x * x
    b6 = y * y
    
    # Gaussian RBF basis with sigma_sq = 2.0
    sigma_sq = 2.0
    
    r1_sq = (x - 0.5)**2 + (y - 0.5)**2
    b7 = np.exp(-sigma_sq * r1_sq)
    
    r2_sq = (x + 0.5)**2 + (y - 0.5)**2
    b8 = np.exp(-sigma_sq * r2_sq)
    
    r3_sq = (x - 0.5)**2 + (y + 0.5)**2
    b9 = np.exp(-sigma_sq * r3_sq)
    
    r4_sq = (x + 0.5)**2 + (y + 0.5)**2
    b10 = np.exp(-sigma_sq * r4_sq)
    
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
