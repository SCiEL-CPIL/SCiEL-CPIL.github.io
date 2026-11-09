"""Symbolic regression with a rich smooth basis: polynomials, sinusoids, Gaussian bump, and tanh."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Flexible model combining polynomial (up to degree 5), sinusoidal,
    Gaussian bump, and tanh terms.

    params (length 10):
        p0: coefficient for constant
        p1: coefficient for x
        p2: coefficient for x^2
        p3: coefficient for x^3
        p4: coefficient for x^4
        p5: amplitude for sin(freq*x)
        p6: frequency for sin/cos terms
        p7: amplitude for cos(freq*x)
        p8: amplitude for Gaussian bump exp(-((x-center)/width)^2)
        p9: center of Gaussian (width fixed at 1.0)
    """
    x = np.asarray(x, dtype=np.float64)

    # Polynomial terms (up to degree 4)
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2

    # Sinusoidal terms with learnable frequency
    freq = params[6]
    freq_clamped = np.clip(freq, -10.0, 10.0)
    sin_term = np.sin(freq_clamped * x)
    cos_term = np.cos(freq_clamped * x)

    # Gaussian bump: exp(-((x - center)^2) / width^2), width=1.0
    center = np.clip(params[9], -5.0, 5.0)
    dx = x - center
    gauss_term = np.exp(-dx * dx)

    # Tanh term
    tanh_term = np.tanh(x)

    # Combine all terms
    result = (
        params[0]
        + params[1] * x
        + params[2] * x2
        + params[3] * x3
        + params[4] * x4
        + params[5] * sin_term
        + params[7] * cos_term
        + params[8] * gauss_term
        + params[9] * tanh_term
    )

    # Ensure finite output
    return np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
# EVOLVE-BLOCK-END
