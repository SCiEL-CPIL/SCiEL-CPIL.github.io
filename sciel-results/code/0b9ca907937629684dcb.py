"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator acceleration model.
    Combines Duffing (cubic position) and Van der Pol (cubic velocity) terms.
    Excludes explicit time dependence to model autonomous dynamics.
    
    y = p0 + p1*x + p2*x^3 + p3*v + p4*v^3
    Unused parameters (p5-p9) are set to 0 to maintain fixed length.
    """
    # Extract parameters
    p0 = params[0]
    p1 = params[1]
    p2 = params[2]
    p3 = params[3]
    p4 = params[4]
    
    # Compute nonlinear terms
    x2 = x * x
    v2 = v * v
    
    # Construct the equation
    # Duffing part: p1*x + p2*x^3
    # Van der Pol part: p3*v + p4*v^3
    # Constant offset: p0
    result = p0 + p1 * x + p2 * (x * x2) + p3 * v + p4 * (v * v2)
    
    return result
# EVOLVE-BLOCK-END
