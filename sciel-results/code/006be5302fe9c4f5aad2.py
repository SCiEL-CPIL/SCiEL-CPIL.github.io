# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form NumPy expression for the acceleration of a nonlinear oscillator.
    
    Uses up to 10 fitted constants (params).
    Physically-motivated structure: linear restoring/damping, cubic stiffness,
    and mixed nonlinear damping terms.
    
    Model (Duffing-van der Pol hybrid):
    a = p0 + p1*x + p2*v + p3*x^3 + p4*x^2*v + p5*v^2 + p6*x^2 + p7*x*v + p8*v^3 + p9*x^4
    """
    x = np.asarray(x, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9]
    
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    v2 = v * v
    v3 = v2 * v
    
    term = (p0 +
            p1 * x +
            p2 * v +
            p3 * x3 +
            p4 * x2 * v +
            p5 * v2 +
            p6 * x2 +
            p7 * x * v +
            p8 * v3 +
            p9 * x4)
            
    return term
# EVOLVE-BLOCK-END
