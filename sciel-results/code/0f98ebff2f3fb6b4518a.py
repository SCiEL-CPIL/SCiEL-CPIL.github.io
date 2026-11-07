"""Symbolic regression: nonlinear oscillator (2) - closed-form equation."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a damped nonlinear oscillator with time-dependent driving force.
    
    Uses a rich nonlinear basis to capture:
    - Linear dynamics (damping, restoring)
    - Nonlinear stiffness (x^2, x^3, x*v)
    - Velocity nonlinearity (v^2, v^3)
    - Time-dependent driving (t, t^2, t*x, t*v)
    - Cross terms
    
    10 fitted constants for a flexible yet stable model.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Core linear terms
    a = p0 * v          # damping
    b = p1 * x          # linear restoring
    
    # Nonlinear stiffness and damping
    c = p2 * x**2       # quadratic stiffness
    d = p3 * x**3       # cubic stiffness (Duffing-like)
    e = p4 * v**2       # quadratic velocity effect
    f = p5 * x * v      # cross nonlinearity
    
    # Time-dependent driving and modulation
    g = p6 * t          # linear time drive
    h = p7 * t * x      # time-modulated restoring
    i = p8 * t * v      # time-modulated damping
    j = p9              # constant offset
    
    return a + b + c + d + e + f + g + h + i + j
# EVOLVE-BLOCK-END
