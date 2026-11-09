"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _legendre_basis(t: np.ndarray) -> np.ndarray:
    """
    Compute shifted Legendre polynomials P0..P3 evaluated at t.
    Returns array of shape (..., 4).
    """
    # t is already normalized to [-1, 1]
    P0 = np.ones_like(t, dtype=np.float64)
    P1 = t
    P2 = 0.5 * (3.0 * t * t - 1.0)
    P3 = 0.5 * (5.0 * t * t * t - 3.0 * t)
    return np.stack([P0, P1, P2, P3], axis=-1)


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression using Legendre polynomial basis.
    
    params has length 10:
      - 3 params for x-basis (P0, P1, P2, P3 -> but we use 3 to save one slot? No, let's be precise)
      
    Actually, let's structure the 10 params clearly:
      We have 3 inputs, each with 4 Legendre terms (P0..P3) = 12 terms. That's too many.
      
    Let's use:
      - P0(x), P1(x), P2(x)  -> 3 terms
      - P0(y), P1(y), P2(y)  -> 3 terms
      - P0(z), P1(z), P2(z)  -> 3 terms
      - P1(x)*P1(y)          -> 1 interaction term
      Total: 10 terms.
      
    params[0:3]   -> coefficients for x Legendre terms P0, P1, P2
    params[3:6]   -> coefficients for y Legendre terms P0, P1, P2
    params[6:9]   -> coefficients for z Legendre terms P0, P1, P2
    params[9]     -> coefficient for P1(x)*P1(y) interaction
    """
    # Normalize inputs to [-1, 1]
    # Inputs are in [0.5, 2.0], center = 1.25, half_range = 0.75
    center = 1.25
    half_range = 0.75
    
    u = (x - center) / half_range
    v = (y - center) / half_range
    w = (z - center) / half_range
    
    # Compute Legendre polynomials up to degree 2 for each input
    # P0(t) = 1
    # P1(t) = t
    # P2(t) = 0.5 * (3t^2 - 1)
    
    Pu0 = np.ones_like(u, dtype=np.float64)
    Pu1 = u
    Pu2 = 0.5 * (3.0 * u * u - 1.0)
    
    Pv0 = np.ones_like(v, dtype=np.float64)
    Pv1 = v
    Pv2 = 0.5 * (3.0 * v * v - 1.0)
    
    Pw0 = np.ones_like(w, dtype=np.float64)
    Pw1 = w
    Pw2 = 0.5 * (3.0 * w * w - 1.0)
    
    p = params
    
    # Linear combination
    result = (
        p[0] * Pu0 + p[1] * Pu1 + p[2] * Pu2 +
        p[3] * Pv0 + p[4] * Pv1 + p[5] * Pv2 +
        p[6] * Pw0 + p[7] * Pw1 + p[8] * Pw2 +
        p[9] * (Pu1 * Pv1)
    )
    
    return result
# EVOLVE-BLOCK-END
