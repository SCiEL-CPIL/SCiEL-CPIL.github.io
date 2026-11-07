"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Polynomial basis with time modulation for nonlinear oscillator.
    Normalized inputs to O(1) scale for well-conditioned BFGS.
    10 fitted weights on a structured basis of (x, v) polynomials modulated by time.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    # Normalize inputs to ~[-1, 1] range
    x_n = (x + 0.015) / 0.19
    v_n = (v - 0.0117) / 0.35
    t_n = (t - 40.0) / 10.0

    # Polynomial features in (x_n, v_n)
    x1 = x_n
    x2 = x_n * x_n
    v1 = v_n
    v2 = v_n * v_n
    xv = x_n * v_n
    x2v = x2 * v_n

    # Time modulation
    sin_t = np.sin(2.0 * np.pi * t_n)
    cos_t = np.cos(2.0 * np.pi * t_n)

    # Basis: p0=const, p1=x, p2=v, p3=x^2, p4=v^2, p5=xv, p6=x^2v,
    #        p7=x*sin(t), p8=v*cos(t), p9=x^2*sin(t)
    result = (
        p0
        + p1 * x1
        + p2 * v1
        + p3 * x2
        + p4 * v2
        + p5 * xv
        + p6 * x2v
        + p7 * x1 * sin_t
        + p8 * v1 * cos_t
        + p9 * x2 * sin_t
    )

    return result
# EVOLVE-BLOCK-END
