"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator model using a physics-informed basis.
    Includes linear and cubic position terms, sine and tanh for bounded nonlinearity,
    linear and cubic velocity terms, and interaction terms.
    """
    # Extract parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    # Precompute powers and nonlinear functions
    x2 = x * x
    x3 = x2 * x
    v2 = v * v
    v3 = v2 * v
    
    sin_x = np.sin(x)
    tanh_x = np.tanh(x)
    
    # Interaction terms
    xv = x * v
    x2v = x2 * v

    # Construct the equation
    # p0: bias
    # p1: linear position
    # p2: cubic position
    # p3: sine position
    # p4: linear velocity
    # p5: cubic velocity
    # p6: interaction x*v
    # p7: interaction x^2*v
    # p8: tanh position
    # p9: quadratic velocity
    
    return (p0 + 
            p1 * x + 
            p2 * x3 + 
            p3 * sin_x + 
            p4 * v + 
            p5 * v3 + 
            p6 * xv + 
            p7 * x2v + 
            p8 * tanh_x + 
            p9 * v2)
# EVOLVE-BLOCK-END
