"""Rewritten program for symbolic regression of a nonlinear oscillator."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a damped nonlinear oscillator with periodic driving force.
    
    Model:
    a = p0 + p1*t + p2*sin(t) + p3*cos(t) + p4*x + p5*x^2 + p6*x^3 + p7*v + p8*v^3 + p9*x*v
    
    Parameters:
    t: Time
    x: Position
    v: Velocity
    params: Array of 10 fitted constants
        p[0]: Bias
        p[1]: Linear time coefficient
        p[2]: Sine drive amplitude/phase component
        p[3]: Cosine drive amplitude/phase component
        p[4]: Linear spring constant
        p[5]: Quadratic spring constant
        p[6]: Cubic spring constant
        p[7]: Linear damping constant
        p[8]: Cubic damping constant
        p[9]: Cross-term (x*v) coefficient
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Pre-compute non-linear features
    sin_t = np.sin(t)
    cos_t = np.cos(t)
    x2 = x * x
    x3 = x2 * x
    v3 = v * v * v
    xv = x * v
    
    # Construct the linear combination of features
    # This structure is linear in parameters, providing a convex optimization landscape for BFGS
    result = (
        p0 
        + p1 * t 
        + p2 * sin_t 
        + p3 * cos_t 
        + p4 * x 
        + p5 * x2 
        + p6 * x3 
        + p7 * v 
        + p8 * v3 
        + p9 * xv
    )
    
    return result
# EVOLVE-BLOCK-END
