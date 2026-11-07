"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator.
    Basis: 1, x, x^2, x^3, v, v^2, x*v, x^2*v, sin(t), cos(t)
    Covers Duffing (x, x^3), Van der Pol damping (v, x^2*v),
    sinusoidal driving (sin(t), cos(t)), plus quadratic and cross terms.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    x2 = x * x
    x3 = x2 * x
    v2 = v * v
    x2v = x2 * v
    xv = x * v
    st = np.sin(t)
    ct = np.cos(t)

    return (
        p0
        + p1 * x
        + p2 * x2
        + p3 * x3
        + p4 * v
        + p5 * v2
        + p6 * xv
        + p7 * x2v
        + p8 * st
        + p9 * ct
    )
# EVOLVE-BLOCK-END
