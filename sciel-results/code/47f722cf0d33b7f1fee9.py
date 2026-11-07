"""Symbolic regression: nonlinear oscillator acceleration."""
# EVOLVE-BLOCK-START
import numpy as np
import warnings

warnings.filterwarnings('ignore')


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form model for the acceleration of a damped nonlinear oscillator.
    Autonomous dynamics (no explicit time dependence).
    Uses a full 3rd-order polynomial basis in position x and velocity v.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    x2 = x * x
    x3 = x2 * x
    v2 = v * v
    v3 = v2 * v

    return (p0
            + p1 * x
            + p2 * v
            + p3 * x2
            + p4 * x * v
            + p5 * v2
            + p6 * x3
            + p7 * x2 * v
            + p8 * x * v2
            + p9 * v3)
# EVOLVE-BLOCK-END
