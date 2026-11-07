"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a damped nonlinear oscillator with driving force, from position and velocity.
    
    Uses a polynomial basis in x and v to capture nonlinear dynamics.
    Basis functions:
      x, v, x^2, x*v, v^2, x^3, x^2*v, x*v^2, v^3, constant
    This allows fitting cubic nonlinearities in position and velocity.
    """
    # Extract parameters
    p0 = params[0]  # coeff for x
    p1 = params[1]  # coeff for v
    p2 = params[2]  # coeff for x^2
    p3 = params[3]  # coeff for x*v
    p4 = params[4]  # coeff for v^2
    p5 = params[5]  # coeff for x^3
    p6 = params[6]  # coeff for x^2*v
    p7 = params[7]  # coeff for x*v^2
    p8 = params[8]  # coeff for v^3
    p9 = params[9]  # constant term
    
    # Compute polynomial terms
    x2 = x * x
    x3 = x2 * x
    v2 = v * v
    v3 = v2 * v
    x_v = x * v
    x2_v = x2 * v
    x_v2 = x * v2
    
    # Linear combination
    return p0 * x + p1 * v + p2 * x2 + p3 * x_v + p4 * v2 + \
           p5 * x3 + p6 * x2_v + p7 * x_v2 + p8 * v3 + p9
# EVOLVE-BLOCK-END
