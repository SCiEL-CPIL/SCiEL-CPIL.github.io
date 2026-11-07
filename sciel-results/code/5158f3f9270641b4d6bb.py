# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form acceleration of a nonlinear damped oscillator.
    Uses up to 10 fitted constants in a structured polynomial basis
    that respects the physics of oscillators (restoring, damping, nonlinear terms).
    
    Structure:
    - Linear restoring: p0 * x
    - Linear damping: p1 * v
    - Constant drive: p2
    - Quadratic restoring: p3 * x^2
    - Quadratic damping: p4 * v^2
    - Cross term: p5 * x * v
    - Cubic restoring: p6 * x^3
    - Cubic damping: p7 * v^3
    - Higher cross: p8 * x^2 * v
    - Higher cross: p9 * x * v^2
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    x2 = x * x
    x3 = x2 * x
    v2 = v * v
    v3 = v2 * v
    xv = x * v
    x2v = x2 * v
    xv2 = x * v2
    
    # Order terms by physical interpretation to help BFGS converge:
    # Restoring force (even powers of x), damping (odd powers of v), drive, cross terms
    return (p0 * x + 
            p1 * v + 
            p2 + 
            p3 * x2 + 
            p4 * v2 + 
            p5 * xv + 
            p6 * x3 + 
            p7 * v3 + 
            p8 * x2v + 
            p9 * xv2)
# EVOLVE-BLOCK-END
