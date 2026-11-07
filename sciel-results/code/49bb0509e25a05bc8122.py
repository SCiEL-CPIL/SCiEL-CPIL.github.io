"""Symbolic regression: nonlinear oscillator acceleration."""
# EVOLVE-BLOCK-START
import numpy as np
import warnings

warnings.filterwarnings('ignore')


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form model for the acceleration of a damped nonlinear oscillator.
    Assumes autonomous, symmetric dynamics.
    Uses a basis of odd polynomial terms and tanh saturation terms.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    x2 = x * x
    x3 = x2 * x
    x5 = x3 * x2
    v2 = v * v
    v3 = v2 * v
    v5 = v3 * v2
    
    tanh_x = np.tanh(x)
    tanh_v = np.tanh(v)

    return (p0
            + p1 * x
            + p2 * v
            + p3 * x3
            + p4 * v3
            + p5 * (x2 * v)
            + p6 * (x * v2)
            + p7 * tanh_x
            + p8 * tanh_v
            + p9 * x5)
# EVOLVE-BLOCK-END
