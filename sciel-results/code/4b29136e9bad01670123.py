"""Symbolic regression: nonlinear oscillator acceleration."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form model for the acceleration of a damped nonlinear oscillator
    with time-dependent driving. Uses up to 10 fitted constants.

    Model: a = p0*t + p1*x + p2*x^3 + p3*v + p4*v*|v| + p5*x*v
           + p6*sin(t) + p7*x^2 + p8*v^2 + p9
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    # Nonlinear restoring force (Duffing-type)
    nonlinear_restoring = p1 * x + p2 * x**3

    # Nonlinear damping (linear + quadratic drag)
    nonlinear_damping = p3 * v + p4 * v * np.abs(v)

    # Coupling term (parametric / gyroscopic)
    coupling = p5 * x * v

    # Time-dependent driving (linear + sinusoidal)
    driving = p0 * t + p6 * np.sin(t)

    # Quadratic corrections
    quadratic = p7 * x**2 + p8 * v**2

    # Offset
    offset = p9

    return nonlinear_restoring + nonlinear_damping + coupling + driving + quadratic + offset
# EVOLVE-BLOCK-END
