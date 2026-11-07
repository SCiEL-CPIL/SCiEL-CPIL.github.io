# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form NumPy expression for the acceleration of a nonlinear oscillator.
    
    Uses up to 10 fitted constants (params).
    Physically-motivated structure with raw inputs (no scaling) to match
    the best-performing configuration (score ~199).
    
    Model:
    a = p0 + p1*x + p2*v + p3*x^2 + p4*v^2 + p5*x*v + p6*x^3 + p7*v^3 + p8*x^2*v + p9*v^2*x
    """
    x = np.asarray(x, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = p
    
    x2 = x * x
    v2 = v * v
    x3 = x2 * x
    v3 = v2 * v
    
    term = (p0 + 
            p1 * x + 
            p2 * v + 
            p3 * x2 + 
            p4 * v2 + 
            p5 * x * v + 
            p6 * x3 + 
            p7 * v3 + 
            p8 * x2 * v + 
            p9 * v2 * x)
            
    return term
# EVOLVE-BLOCK-END
