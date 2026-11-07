"""Rewritten program: symbolic regression for nonlinear oscillator with physics-based basis."""
import numpy as np


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a damped nonlinear oscillator with driving force.
    
    Uses a Van der Pol-inspired polynomial basis in position x and velocity v.
    Basis functions (10 terms total):
      1. x          (linear restoring force)
      2. v          (linear damping)
      3. x^2        (quadratic nonlinearity in position)
      4. x*v        (cross term)
      5. v^2        (quadratic nonlinearity in velocity)
      6. x^3        (cubic restoring, Van der Pol)
      7. x^2*v      (cubic cross)
      8. x*v^2      (cubic cross)
      9. v^3        (cubic damping)
      10. constant  (bias)
    
    This captures the typical nonlinear oscillator dynamics where the 
    restoring force has cubic terms (Van der Pol) and damping may be 
    nonlinear in velocity.
    """
    p0 = params[0]
    p1 = params[1]
    p2 = params[2]
    p3 = params[3]
    p4 = params[4]
    p5 = params[5]
    p6 = params[6]
    p7 = params[7]
    p8 = params[8]
    p9 = params[9]
    
    x2 = x * x
    x3 = x2 * x
    v2 = v * v
    v3 = v2 * v
    
    return (p0 * x + p1 * v + p2 * x2 + p3 * x * v + p4 * v2 +
            p5 * x3 + p6 * x2 * v + p7 * x * v2 + p8 * v3 + p9)
