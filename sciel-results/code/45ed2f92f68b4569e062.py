import numpy as np

# EVOLVE-BLOCK-START

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Duffing-type nonlinear oscillator:
    a = p0 + p1*x + p2*x^2 + p3*x^3 + p4*v + p5*v^2 + p6*x*v
      + p7*cos(t) + p8*sin(t) + p9*sin(2*t)
    
    This encodes the physics: cubic stiffness (Duffing), quadratic damping,
    cross-coupling, and periodic driving force.
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
    
    # Polynomial terms in x and v
    x2 = x * x
    x3 = x2 * x
    v2 = v * v
    xv = x * v
    
    # Driving force terms (periodic)
    drive = p7 * np.cos(t) + p8 * np.sin(t) + p9 * np.sin(2.0 * t)
    
    pred = p0 + p1*x + p2*x2 + p3*x3 + p4*v + p5*v2 + p6*xv + drive
    
    return pred

# EVOLVE-BLOCK-END
