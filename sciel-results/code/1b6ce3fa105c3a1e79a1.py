"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator modeled as a Van der Pol type system.
    
    Uses a physically-motivated nonlinear basis that captures:
    - Linear restoring and damping (x, v)
    - Amplitude-dependent damping (x^2*v, v^3) characteristic of limit cycles
    - Cubic nonlinearity in position (x^3)
    - Cross terms (x*v, v^2, x^2, x^4, x^5)
    
    The basis is designed so that BFGS can find meaningful coefficients
    for the limit-cycle dynamics where energy is injected at small amplitudes
    and dissipated at large amplitudes.
    """
    p0 = params[0]   # coeff for x (linear restoring)
    p1 = params[1]   # coeff for v (linear damping)
    p2 = params[2]   # coeff for x^2
    p3 = params[3]   # coeff for x*v
    p4 = params[4]   # coeff for v^2
    p5 = params[5]   # coeff for x^3 (cubic nonlinearity)
    p6 = params[6]   # coeff for x^2*v (amplitude-dependent damping)
    p7 = params[7]   # coeff for x*v^2
    p8 = params[8]   # coeff for v^3 (velocity-dependent damping)
    p9 = params[9]   # constant / bias
    
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    x5 = x4 * x
    v2 = v * v
    v3 = v2 * v
    
    return (p0 * x + p1 * v + p2 * x2 + p3 * x * v + p4 * v2 +
            p5 * x3 + p6 * x2 * v + p7 * x * v2 + p8 * v3 + p9)
# EVOLVE-BLOCK-END
