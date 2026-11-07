"""Symbolic regression for a nonlinear oscillator."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator with Duffing, Van der Pol, and sinusoidal driving.
    Form: p0 + p1*x + p2*x^3 + p3*v + p4*v^3 + p5*x*v + p6*sin(t) + p7*cos(t) + p8*t + p9*x^2
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    result = (
        p0
        + p1 * x
        + p2 * x**3
        + p3 * v
        + p4 * v**3
        + p5 * x * v
        + p6 * np.sin(t)
        + p7 * np.cos(t)
        + p8 * t
        + p9 * x**2
    )
    
    return result
# EVOLVE-BLOCK-END
