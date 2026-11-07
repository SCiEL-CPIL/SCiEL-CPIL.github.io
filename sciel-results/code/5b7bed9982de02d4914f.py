import numpy as np
import time

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Trigonometric-hybrid nonlinear oscillator model.
    
    Uses sinusoidal basis functions to capture the periodic nature of oscillator
    dynamics, combined with a few polynomial terms for amplitude-dependent effects.
    This avoids conditioning issues of high-degree polynomials and directly encodes
    the physics of nonlinear oscillation.
    
    Basis:
    0: 1 (offset)
    1: sin(x) - primary restoring force (linear for small x)
    2: v - linear damping
    3: sin(x)*x - nonlinear restoring (cubic-like for small x)
    4: v^2 - amplitude-dependent damping (Van der Pol)
    5: sin(x)*v - cross coupling
    6: sin(2*x) - second harmonic restoring
    7: v^3 - nonlinear damping
    8: x^2 - even-order displacement effect
    9: cos(x)*v - phase-dependent damping
    """
    p = params
    
    # Use raw inputs (small range, well-conditioned for trig functions)
    # x in [-0.5728, 0.5275], v in [-0.1754, 0.2821]
    # These are small enough that trig functions behave well
    
    sin_x = np.sin(x)
    cos_x = np.cos(x)
    sin_2x = np.sin(2.0 * x)
    
    v2 = v * v
    v3 = v2 * v
    x2 = x * x
    
    return (p[0] +
            p[1] * sin_x +
            p[2] * v +
            p[3] * sin_x * x +
            p[4] * v2 +
            p[5] * sin_x * v +
            p[6] * sin_2x +
            p[7] * v3 +
            p[8] * x2 +
            p[9] * cos_x * v)
