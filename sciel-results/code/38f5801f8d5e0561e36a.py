"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 140.0

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a damped nonlinear oscillator.
    Uses a Duffing-like structure with separate spring and damping terms,
    plus higher-order corrections, to capture nonlinear oscillator dynamics.
    
    Model: a = p[0]*(k1*x + k3*x^3) + p[1]*(c1*v + c3*v^3) 
           + p[2]*x^2*v + p[3]*x*v^2 + p[4]*x^2 + p[5]*v^2
           + p[6]*x^4 + p[7]*v^4 + p[8]*x^3*v + p[9]*x*v^3
    
    This structure separates the main restoring force (Duffing) from
    damping (cubic), which matches the physics of nonlinear oscillators.
    """
    p = params
    
    # Precompute powers
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    v2 = v * v
    v3 = v2 * v
    v4 = v2 * v2
    
    # Duffing spring term: linear + cubic restoring force
    spring = p[0] * (x + p[1] * x3)
    
    # Cubic damping term: linear + cubic velocity damping
    damping = p[2] * (v + p[3] * v3)
    
    # Cross terms for more general nonlinear coupling
    cross1 = p[4] * x2 * v
    cross2 = p[5] * x * v2
    cross3 = p[6] * x3 * v
    cross4 = p[7] * x * v3
    
    # Quadratic corrections
    quad_x = p[8] * x2
    quad_v = p[9] * v2
    
    return spring + damping + cross1 + cross2 + cross3 + cross4 + quad_x + quad_v
# EVOLVE-BLOCK-END
