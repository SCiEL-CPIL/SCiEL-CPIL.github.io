"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a damped nonlinear oscillator using a cubic polynomial basis.
    Includes linear, quadratic, and cubic terms for position and velocity,
    as well as interaction terms, to capture nonlinear restoring and damping forces.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Precompute powers
    x2 = x * x
    x3 = x2 * x
    
    v2 = v * v
    v3 = v2 * v
    
    # Interaction terms
    xv = x * v
    x2v = x2 * v
    xv2 = x * v2
    
    # Construct the equation
    # Basis: 1, x, v, x^2, x^3, v^2, v^3, x*v, x^2*v, x*v^2
    return p0 + p1 * x + p2 * v + p3 * x2 + p4 * x3 + p5 * v2 + p6 * v3 + p7 * xv + p8 * x2v + p9 * xv2
# EVOLVE-BLOCK-END
