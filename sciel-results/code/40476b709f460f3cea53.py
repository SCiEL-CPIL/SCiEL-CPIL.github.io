"""Symbolic regression: nonlinear oscillator acceleration."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form model for the acceleration of a damped nonlinear oscillator
    with Van der Pol-style negative damping and cubic velocity saturation,
    Duffing-type position nonlinearity, and light time-dependent driving.
    Uses up to 10 fitted constants.

    Model:
      a = p0 * x
        + p1 * x^3            (Duffing cubic restoring)
        + p2 * v              (linear damping; can be negative -> Van der Pol)
        + p3 * v^3            (cubic velocity saturation)
        + p4 * x^2 * v        (Van der Pol nonlinear damping)
        + p5 * x * v          (parametric / gyroscopic coupling)
        + p6 * t              (slow linear drift / secular term)
        + p7 * sin(t)         (sinusoidal drive)
        + p8 * x^2            (quadratic position correction)
        + p9                  (offset)
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    # Duffing-type restoring (linear + cubic)
    restoring = p0 * x + p1 * x ** 3

    # Van der Pol style damping: linear (can be negative) + cubic saturation + x^2*v
    damping = p2 * v + p3 * v ** 3 + p4 * x ** 2 * v

    # Parametric / gyroscopic coupling
    coupling = p5 * x * v

    # Light time-dependent terms (drift + sinusoidal drive)
    time_terms = p6 * t + p7 * np.sin(t)

    # Quadratic position correction and offset
    correction = p8 * x ** 2 + p9

    return restoring + damping + coupling + time_terms + correction
# EVOLVE-BLOCK-END
