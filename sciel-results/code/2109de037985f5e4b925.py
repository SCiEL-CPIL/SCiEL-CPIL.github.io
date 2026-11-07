"""Symbolic regression candidate: normalized quadratic basis."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Quadratic polynomial in normalized inputs.
    Inputs are in [0.5, 2]. We normalize to [-1, 1] via:
        u = (v - 1.25) / 0.75
    This improves conditioning for BFGS while keeping the same functional form.

    Basis (10 terms):
    0: 1
    1: u_x
    2: u_y
    3: u_z
    4: u_x^2
    5: u_y^2
    6: u_z^2
    7: u_x * u_y
    8: u_x * u_z
    9: u_y * u_z
    """
    p = params

    # Normalize to [-1, 1]
    ux = (x - 1.25) / 0.75
    uy = (y - 1.25) / 0.75
    uz = (z - 1.25) / 0.75

    return (p[0] +
            p[1] * ux +
            p[2] * uy +
            p[3] * uz +
            p[4] * ux * ux +
            p[5] * uy * uy +
            p[6] * uz * uz +
            p[7] * ux * uy +
            p[8] * ux * uz +
            p[9] * uy * uz)
# EVOLVE-BLOCK-END
