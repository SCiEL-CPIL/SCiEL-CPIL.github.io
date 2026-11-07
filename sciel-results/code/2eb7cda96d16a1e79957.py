"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator acceleration model.
    
    Combines:
    - Duffing nonlinearity: x^3
    - Van der Pol damping: v^3
    - Quadratic coupling: x^2, v^2, x*v
    - Sinusoidal driving: sin(t), cos(t)
    - Linear terms: x, v, constant
    
    10 parameters total.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Precompute common terms
    x2 = x * x
    v2 = v * v
    x3 = x2 * x
    v3 = v2 * v
    
    # Model: constant + linear + nonlinear + driving
    result = (
        p0                          # constant
        + p1 * x                    # linear position
        + p2 * v                    # linear velocity (damping)
        + p3 * x3                   # Duffing nonlinearity
        + p4 * v3                   # Van der Pol nonlinearity
        + p5 * x2                   # quadratic position
        + p6 * v2                   # quadratic velocity
        + p7 * x * v                # cross term
        + p8 * np.sin(t)            # sinusoidal driving
        + p9 * np.cos(t)            # phase-shifted driving
    )
    
    return result
# EVOLVE-BLOCK-END
