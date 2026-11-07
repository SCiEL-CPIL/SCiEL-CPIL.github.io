"""Symbolic regression: nonlinear oscillator acceleration."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form model for the acceleration of a damped nonlinear oscillator.
    
    Basis functions:
    1. x (linear restoring)
    2. x^3 (cubic restoring, Duffing)
    3. v (linear damping)
    4. v^3 (cubic damping)
    5. x^2 * v (Van der Pol damping)
    6. sin(t) (sinusoidal drive)
    7. cos(t) (cosinusoidal drive / phase shift)
    8. x * sin(t) (parametric excitation)
    9. v * sin(t) (velocity-dependent drive modulation)
    10. 1 (offset)
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    # Restoring forces
    linear_restoring = p0 * x
    cubic_restoring = p1 * x**3

    # Damping terms
    linear_damping = p2 * v
    cubic_damping = p3 * v**3
    vdp_damping = p4 * x**2 * v  # Van der Pol type

    # Driving terms
    sin_drive = p5 * np.sin(t)
    cos_drive = p6 * np.cos(t)

    # Coupling / parametric terms
    x_sin = p7 * x * np.sin(t)
    v_sin = p8 * v * np.sin(t)

    # Offset
    offset = p9

    return (linear_restoring + cubic_restoring + linear_damping + 
            cubic_damping + vdp_damping + sin_drive + cos_drive + 
            x_sin + v_sin + offset)
# EVOLVE-BLOCK-END
