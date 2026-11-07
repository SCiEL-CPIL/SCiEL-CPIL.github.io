"""Symbolic regression: nonlinear oscillator acceleration."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form model for the acceleration of a nonlinear oscillator.
    
    Structural change: Simplified polynomial basis to improve BFGS convergence.
    Removed pure cubic terms (xn^3, vn^3) which can cause ill-conditioning 
    if the signal is not strongly cubic, and replaced with mixed terms 
    (xn*vn, xn^2, vn^2, xn^2*vn, vn^2*xn) which are often more relevant 
    in coupled oscillator dynamics (Van der Pol, Duffing with damping).
    
    Normalization is kept to ensure the starting point (params=ones) 
    is close to the optimal scale for linear terms.
    
    Model (in normalized variables):
      a = p0 * xn
        + p1 * vn
        + p2 * xn * vn
        + p3 * xn^2
        + p4 * vn^2
        + p5 * xn^2 * vn
        + p6 * vn^2 * xn
        + p7 * tn
        + p8 * sin(t)
        + p9
    
    Note: sin(t) uses raw t to preserve the physical oscillation frequency.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    # Normalize inputs to improve conditioning
    tn = (t - 40.0) / 10.0
    xn = x / 0.2
    vn = v / 0.35

    # Precompute powers
    xn2 = xn * xn
    vn2 = vn * vn

    # Linear terms
    linear = p0 * xn + p1 * vn

    # Quadratic / Mixed terms
    quadratic = p2 * (xn * vn) + p3 * xn2 + p4 * vn2

    # Cubic / Mixed terms (no pure cubes)
    cubic = p5 * (xn2 * vn) + p6 * (vn2 * xn)

    # Time-dependent terms
    time_terms = p7 * tn + p8 * np.sin(t)

    # Offset
    offset = p9

    return linear + quadratic + cubic + time_terms + offset
# EVOLVE-BLOCK-END
