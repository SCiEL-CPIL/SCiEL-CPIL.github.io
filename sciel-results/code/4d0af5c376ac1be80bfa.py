"""Symbolic regression: nonlinear oscillator acceleration."""
# EVOLVE-BLOCK-START
import numpy as np
import warnings

warnings.filterwarnings('ignore')


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form model for the acceleration of a damped nonlinear oscillator.
    Uses a full 3rd-order polynomial basis in position x and velocity v.
    The model is autonomous (no explicit time dependence) since the dynamics
    of a nonlinear oscillator are typically autonomous.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    # Compute powers
    x2 = x * x
    x3 = x2 * x
    v2 = v * v
    v3 = v2 * v

    # Full 3rd-order polynomial basis in x and v
    # Terms: 1, x, v, x^2, x*v, v^2, x^3, x^2*v, x*v^2, v^3
    term0 = p0          # constant
    term1 = p1 * x      # linear in x
    term2 = p2 * v      # linear in v
    term3 = p3 * x2     # quadratic in x
    term4 = p4 * x * v  # cross term
    term5 = p5 * v2     # quadratic in v
    term6 = p6 * x3     # cubic in x
    term7 = p7 * x2 * v # x^2 * v
    term8 = p8 * x * v2 # x * v^2
    term9 = p9 * v3     # cubic in v

    return term0 + term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 + term9
# EVOLVE-BLOCK-END
