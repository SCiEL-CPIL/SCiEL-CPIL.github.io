"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a damped nonlinear oscillator (Van der Pol type) with driving force.
    
    Uses a physics-informed basis that captures:
    - Linear restoring force (x)
    - Linear damping (v)
    - Nonlinear restoring (x^3)
    - Amplitude-dependent damping (v^3, x^2*v)
    - Cross terms (x*v, x^2, v^2)
    - Higher order terms (x^5, v^2*x, x^4)
    - Constant (driving)
    
    Basis: {1, x, v, x^2, x*v, v^2, x^3, x^2*v, x*v^2, v^3}
    This 10-term basis captures limit-cycle dynamics where energy is injected
    at small amplitudes (negative effective damping) and dissipated at large
    amplitudes (positive effective damping).
    """
    p0 = params[0]   # constant (driving offset)
    p1 = params[1]   # x (linear restoring)
    p2 = params[2]   # v (linear damping)
    p3 = params[3]   # x^2 (quadratic nonlinearity)
    p4 = params[4]   # x*v (cross term)
    p5 = params[5]   # v^2 (quadratic velocity)
    p6 = params[6]   # x^3 (cubic restoring - Van der Pol)
    p7 = params[7]   # x^2*v (amplitude-dependent damping)
    p8 = params[8]   # x*v^2 (cross nonlinearity)
    p9 = params[9]   # v^3 (cubic damping - Van der Pol)
    
    x2 = x * x
    v2 = v * v
    x3 = x2 * x
    v3 = v2 * v
    
    return (p0 
            + p1 * x 
            + p2 * v 
            + p3 * x2 
            + p4 * x * v 
            + p5 * v2 
            + p6 * x3 
            + p7 * x2 * v 
            + p8 * x * v2 
            + p9 * v3)
# EVOLVE-BLOCK-END
