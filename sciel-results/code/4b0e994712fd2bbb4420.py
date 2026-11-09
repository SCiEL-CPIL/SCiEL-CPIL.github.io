"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def _legendre_p0(t):
    return np.ones_like(t)

def _legendre_p1(t):
    return t

def _legendre_p2(t):
    return 0.5 * (3 * t**2 - 1)

def _legendre_p3(t):
    return 0.5 * (5 * t**3 - 3 * t)


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using shifted Legendre polynomials of degree up to 3
    applied to each normalized input, plus one cross-interaction term.
    
    Inputs x, y, z are in [0.5, 2]. We normalize them to [-1, 1] for Legendre:
    t = (2 * input - 2.5) / 1.5  maps [0.5, 2] -> [-1, 1]
    
    Basis terms:
    - P0, P1, P2, P3 for each of x, y, z (12 terms)
    - Constant (already covered by P0 terms, but we keep explicit constant via params[0] scaling)
    - Cross-interaction: P1(x_norm) * P1(y_norm)
    
    Total: 12 + 1 = 13 basis functions, but params has length 10.
    We need to select 10 basis functions.
    
    Let's use:
    0: constant (P0 for x, which is 1)
    1: P1(x_norm)
    2: P2(x_norm)
    3: P3(x_norm)
    4: P1(y_norm)
    5: P2(y_norm)
    6: P3(y_norm)
    7: P1(z_norm)
    8: P2(z_norm)
    9: P1(x_norm) * P1(y_norm)  (cross-interaction)
    
    This gives us 10 terms. We omit P0 for y and z (they're all 1, redundant with constant)
    and omit P3(z) to stay within 10 params.
    """
    # Normalize inputs to [-1, 1]
    # Input range [0.5, 2] -> center 1.25, half-range 0.75
    # t = (input - 1.25) / 0.75
    x_t = (x - 1.25) / 0.75
    y_t = (y - 1.25) / 0.75
    z_t = (z - 1.25) / 0.75
    
    # Compute Legendre polynomials
    x_p0 = _legendre_p0(x_t)
    x_p1 = _legendre_p1(x_t)
    x_p2 = _legendre_p2(x_t)
    x_p3 = _legendre_p3(x_t)
    
    y_p1 = _legendre_p1(y_t)
    y_p2 = _legendre_p2(y_t)
    y_p3 = _legendre_p3(y_t)
    
    z_p1 = _legendre_p1(z_t)
    z_p2 = _legendre_p2(z_t)
    
    # Cross-interaction
    cross = x_p1 * y_p1
    
    # Combine with params (length 10)
    result = (params[0] * x_p0 +
              params[1] * x_p1 +
              params[2] * x_p2 +
              params[3] * x_p3 +
              params[4] * y_p1 +
              params[5] * y_p2 +
              params[6] * y_p3 +
              params[7] * z_p1 +
              params[8] * z_p2 +
              params[9] * cross)
    
    return result
# EVOLVE-BLOCK-END
