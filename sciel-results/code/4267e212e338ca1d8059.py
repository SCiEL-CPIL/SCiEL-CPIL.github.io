"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# The evaluator calls equation(t, x, v, params).
# We define a robust nonlinear oscillator model.
# Structure: Coupled time-state terms (parametric drive) + Nonlinear spring/damping.
# This covers Mathieu-type parametric excitation and Van der Pol damping.

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression for a nonlinear oscillator acceleration.
    
    y = C(t, x, v)
    
    C(t, x, v) = p0*sin(t)*x + p1*cos(t)*x + p2*sin(t)*v + p3*cos(t)*v
                + p4*x + p5*x^2 + p6*x^3 + p7*v + p8*x*v + p9*x^2*v
    
    Total 10 parameters.
    """
    # Pre-compute trigonometric functions
    sin_t = np.sin(t)
    cos_t = np.cos(t)
    
    # Polynomial/Interaction terms
    x2 = x * x
    x3 = x2 * x
    xv = x * v
    x2v = x2 * v
    
    # Combine with parameters
    # p0, p1: Parametric drive on x
    # p2, p3: Parametric drive on v
    # p4, p5, p6: Spring force (Linear, Quadratic, Cubic)
    # p7: Linear damping
    # p8, p9: Nonlinear damping (x*v, x^2*v)
    
    y = (params[0] * sin_t * x + params[1] * cos_t * x + 
         params[2] * sin_t * v + params[3] * cos_t * v +
         params[4] * x + params[5] * x2 + params[6] * x3 +
         params[7] * v + params[8] * xv + params[9] * x2v)
         
    return y
# EVOLVE-BLOCK-END
