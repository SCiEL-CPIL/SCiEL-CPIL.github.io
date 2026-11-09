import numpy as np


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Physics-informed Duffing-van der Pol oscillator model.
    
    Structure: a = p0*x + p1*x^3 - p2*v - p3*v^3 + p4*x*v^2 + p5*v^2*x^2 
               + p6*x^2*v + p7*x^4 + p8*v^4 + p9
    
    Parameters:
      p0: linear spring coefficient
      p1: cubic stiffness (Duffing)
      p2: linear damping
      p3: cubic damping (van der Pol)
      p4: cross term x*v^2
      p5: cross term v^2*x^2
      p6: cross term x^2*v
      p7: quartic stiffness
      p8: quartic damping
      p9: bias
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
    x4 = x2 * x2
    
    v2 = v * v
    v3 = v2 * v
    v4 = v2 * v2
    
    # Cross terms
    xv2 = x * v2
    v2x2 = v2 * x2
    x2v = x2 * v
    
    result = (p0 * x + p1 * x3 - p2 * v - p3 * v3 
              + p4 * xv2 + p5 * v2x2 + p6 * x2v 
              + p7 * x4 + p8 * v4 + p9)
    
    return result
