"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a damped nonlinear oscillator (Duffing-van der Pol style).
    
    Structure: a = p1*x + p2*x^3 - p3*v - p4*v^3 + p5*x*v^2 + p6*v^2*x^2 
               + p7*x^2*v + p8*x^4 + p9*v^4 + p10
    
    Inputs:
        x: position (array)
        v: velocity (array)
        params: array of 10 fitted constants
    """
    # Compute powers
    x2 = x * x
    v2 = v * v
    x3 = x2 * x
    v3 = v2 * v
    x4 = x2 * x2
    v4 = v2 * v2
    
    # Construct terms
    # p1: linear position (spring)
    # p2: cubic position (Duffing nonlinearity)
    # p3: linear velocity (damping)
    # p4: cubic velocity (van der Pol nonlinearity)
    # p5: x * v^2 (coupling)
    # p6: x^2 * v^2 (quadratic coupling)
    # p7: x^2 * v (position-squared damping)
    # p8: x^4 (quartic position)
    # p9: v^4 (quartic velocity)
    # p10: constant offset
    
    term1 = x
    term2 = x3
    term3 = v
    term4 = v3
    term5 = x * v2
    term6 = x2 * v2
    term7 = x2 * v
    term8 = x4
    term9 = v4
    term10 = np.ones_like(x)
    
    return (params[0] * term1 +
            params[1] * term2 +
            params[2] * term3 +
            params[3] * term4 +
            params[4] * term5 +
            params[5] * term6 +
            params[6] * term7 +
            params[7] * term8 +
            params[8] * term9 +
            params[9] * term10)
# EVOLVE-BLOCK-END
