"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a damped nonlinear oscillator (Duffing type) with driving force.
    
    Model: a = c0 + c1*x + c2*x^2 + c3*x^3 + c4*v + c5*v^2 + c6*v^3 + c7*x*v + c8*x^2*v + c9*v^2*x
    
    This captures:
    - Linear restoring (c1*x)
    - Nonlinear restoring/Duffing (c3*x^3, c2*x^2)
    - Linear damping (c4*v)
    - Nonlinear damping (c5*v^2, c6*v^3)
    - Coupling terms (c7*x*v, c8*x^2*v, c9*v^2*x)
    
    Inputs:
        x: position array
        v: velocity array
        params: 10 fitted constants
    
    Returns:
        Acceleration predictions
    """
    c0, c1, c2, c3, c4, c5, c6, c7, c8, c9 = params
    
    x2 = x * x
    x3 = x2 * x
    v2 = v * v
    v3 = v2 * v
    
    # Build the full cubic polynomial basis
    # Order: 1, x, x^2, x^3, v, v^2, v^3, x*v, x^2*v, v^2*x
    term0 = c0
    term1 = c1 * x
    term2 = c2 * x2
    term3 = c3 * x3
    term4 = c4 * v
    term5 = c5 * v2
    term6 = c6 * v3
    term7 = c7 * x * v
    term8 = c8 * x2 * v
    term9 = c9 * v2 * x
    
    return term0 + term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 + term9
# EVOLVE-BLOCK-END
