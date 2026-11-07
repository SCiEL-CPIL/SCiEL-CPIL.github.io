"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator acceleration model using Van der Pol-inspired basis.
    Unscaled inputs for better BFGS convergence (matching parent's success).
    Basis: {1, x, v, x^3, v^2, xv, x^5, v^3, x^2v, x^4}
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    x5 = x4 * x
    v2 = v * v
    v3 = v2 * v
    xv = x * v
    x2v = x2 * v
    
    return (p0 + 
            p1 * x + 
            p2 * v + 
            p3 * x3 + 
            p4 * v2 + 
            p5 * xv + 
            p6 * x5 + 
            p7 * v3 + 
            p8 * x2v + 
            p9 * x4)
# EVOLVE-BLOCK-END
