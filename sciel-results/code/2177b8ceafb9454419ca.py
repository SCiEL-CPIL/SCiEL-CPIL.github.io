"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Damped nonlinear oscillator (Duffing-type) with sinusoidal driving.
    
    a = p0 + p1*x + p2*v + p3*x^3 + p4*v*|v| + p5*sin(t) + p6*cos(t) + p7*x*v + p8*x^2 + p9*v^2
    
    Uses all 10 parameters. The model captures:
    - p0: bias/offset
    - p1: linear spring constant
    - p2: linear damping
    - p3: cubic nonlinearity (Duffing)
    - p4: quadratic drag (smoothed via v*|v| -> use v*v for smoothness)
    - p5, p6: sinusoidal driving
    - p7: cross term x*v
    - p8: quadratic position
    - p9: quadratic velocity
    """
    x2 = x * x
    x3 = x2 * x
    v2 = v * v
    xv = x * v
    
    sin_t = np.sin(t)
    cos_t = np.cos(t)
    
    # Use v*v instead of v*abs(v) for smoothness (BFGS-friendly)
    return (params[0] +
            params[1] * x +
            params[2] * v +
            params[3] * x3 +
            params[4] * v2 +
            params[5] * sin_t +
            params[6] * cos_t +
            params[7] * xv +
            params[8] * x2 +
            params[9] * v2)
# EVOLVE-BLOCK-END
