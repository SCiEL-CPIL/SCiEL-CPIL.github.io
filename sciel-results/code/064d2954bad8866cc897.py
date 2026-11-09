"""Rewritten program for symbolic regression with 3 inputs."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _legendre_shifted(n, t):
    """
    Compute the n-th Legendre polynomial evaluated at t, shifted to domain [0.5, 2].
    We map t in [0.5, 2] to u in [-1, 1] via u = (t - 1.25) / 0.75.
    Then P_n(u) is the standard Legendre polynomial.
    We use the recurrence relation for numerical stability.
    """
    # Map t from [0.5, 2] to u in [-1, 1]
    # center = (0.5 + 2) / 2 = 1.25, half_width = (2 - 0.5) / 2 = 0.75
    u = (t - 1.25) / 0.75
    
    # Clamp u to [-1, 1] for safety, though inputs are guaranteed in [0.5, 2]
    u = np.clip(u, -1.0, 1.0)
    
    if n == 0:
        return np.ones_like(u)
    elif n == 1:
        return u
    
    P0 = np.ones_like(u)
    P1 = u.copy()
    for k in range(1, n):
        P2 = ((2 * k + 1) * u * P1 - k * P0) / (k + 1)
        P0 = P1
        P1 = P2
    return P1


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Non-linear parametric model: Product of two linear forms plus correction terms.
    This allows the model to capture multiplicative interactions and quadratic surfaces
    more flexibly than a fixed linear basis.
    
    y = (p0 + p1*x + p2*y + p3*z) * (p4 + p5*x + p6*y + p7*z) + p8*(x*y*z) + p9*(x^2 + y^2 + z^2)
    """
    p = params
    
    # Linear form 1
    L1 = p[0] + p[1] * x + p[2] * y + p[3] * z
    
    # Linear form 2
    L2 = p[4] + p[5] * x + p[6] * y + p[7] * z
    
    # Product term
    term_product = L1 * L2
    
    # Correction terms to allow for non-rank-2 quadratic parts and cubic interaction
    term_cubic = p[8] * (x * y * z)
    term_quad = p[9] * (x**2 + y**2 + z**2)
    
    pred = term_product + term_cubic + term_quad
    
    return pred
# EVOLVE-BLOCK-END
