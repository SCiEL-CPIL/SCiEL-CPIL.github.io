"""Symbolic regression: nonlinear oscillator acceleration."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form model for the acceleration of a damped nonlinear oscillator.
    
    Refined Basis Functions:
    1. 1 (offset)
    2. x (linear restoring)
    3. x^3 (cubic restoring, Duffing)
    4. v (linear damping)
    5. x^2 * v (Van der Pol damping)
    6. x * v (nonlinear damping cross-term)
    7. sin(t) (sinusoidal drive)
    8. cos(t) (cosinusoidal drive)
    9. x * cos(t) (parametric excitation, cosine phase)
    10. x * sin(t) (parametric excitation, sine phase)
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    # Restoring forces
    linear_restoring = p0 * x
    cubic_restoring = p1 * x**3

    # Damping terms
    linear_damping = p2 * v
    vdp_damping = p3 * x**2 * v
    cross_damping = p4 * x * v

    # Driving terms
    sin_drive = p5 * np.sin(t)
    cos_drive = p6 * np.cos(t)

    # Parametric excitation terms
    x_cos = p7 * x * np.cos(t)
    x_sin = p8 * x * np.sin(t)

    # Offset
    offset = p9

    return (linear_restoring + cubic_restoring + linear_damping + 
            vdp_damping + cross_damping + sin_drive + cos_drive + 
            x_cos + x_sin + offset)
# EVOLVE-BLOCK-END
