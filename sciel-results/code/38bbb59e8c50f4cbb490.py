"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid basis: orthogonal Legendre polynomials for polynomial trends,
    plus trigonometric and non-polynomial basis functions for local structure.
    
    params[0]: constant (P0)
    params[1]: linear (P1)
    params[2]: quadratic (P2)
    params[3]: cubic (P3)
    params[4]: quartic (P4)
    params[5]: sin(x)
    params[6]: cos(x)
    params[7]: sin(2x)
    params[8]: tanh(x)
    params[9]: exp(-x^2)
    
    Using Legendre polynomials (orthogonal on [-1,1]) improves conditioning
    compared to monomials, allowing BFGS to converge more reliably.
    """
    x = np.asarray(x, dtype=np.float64)
    p = params

    # Scale x to [-1, 1] for Legendre polynomials (x is in [-3, 3])
    u = x / 3.0

    # Legendre polynomials P0..P4
    # P0 = 1
    # P1 = u
    # P2 = (3u^2 - 1) / 2
    # P3 = (5u^3 - 3u) / 2
    # P4 = (35u^4 - 30u^2 + 3) / 8
    u2 = u * u
    u3 = u2 * u
    u4 = u2 * u2

    P0 = np.ones_like(x)
    P1 = u
    P2 = 0.5 * (3.0 * u2 - 1.0)
    P3 = 0.5 * (5.0 * u3 - 3.0 * u)
    P4 = 0.125 * (35.0 * u4 - 30.0 * u2 + 3.0)

    # Trigonometric and non-polynomial terms
    sx = np.sin(x)
    cx = np.cos(x)
    s2x = np.sin(2.0 * x)
    tx = np.tanh(x)
    gx = np.exp(-x * x)

    result = (
        p[0] * P0
        + p[1] * P1
        + p[2] * P2
        + p[3] * P3
        + p[4] * P4
        + p[5] * sx
        + p[6] * cx
        + p[7] * s2x
        + p[8] * tx
        + p[9] * gx
    )

    return result
# EVOLVE-BLOCK-END
