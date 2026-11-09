"""Symbolic regression with shifted Legendre polynomial basis."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Normalize inputs from [0.5, 2.0] to [-1, 1]
    u = (x - 1.25) / 0.75
    v = (y - 1.25) / 0.75
    w = (z - 1.25) / 0.75
    
    p = params
    
    # Shifted Legendre polynomials on [-1, 1]
    # P0(t) = 1
    # P1(t) = t
    # P2(t) = 0.5 * (3t^2 - 1)
    # P3(t) = 0.5 * (5t^3 - 3t)
    
    Pu1 = u
    Pu2 = 0.5 * (3.0 * u * u - 1.0)
    Pu3 = 0.5 * (5.0 * u * u * u - 3.0 * u)
    
    Pv1 = v
    Pv2 = 0.5 * (3.0 * v * v - 1.0)
    
    Pw1 = w
    Pw2 = 0.5 * (3.0 * w * w - 1.0)
    
    # 10 basis terms mapped to 10 params
    # params[0]: constant (P0)
    # params[1]: P1(x) - linear in x
    # params[2]: P1(y) - linear in y
    # params[3]: P1(z) - linear in z
    # params[4]: P2(x) - quadratic in x
    # params[5]: P2(y) - quadratic in y
    # params[6]: P2(z) - quadratic in z
    # params[7]: P1(x)*P1(y) - interaction x*y
    # params[8]: P1(x)*P1(z) - interaction x*z
    # params[9]: P3(x) - cubic in x
    
    result = (
        p[0] * 1.0
        + p[1] * Pu1
        + p[2] * Pv1
        + p[3] * Pw1
        + p[4] * Pu2
        + p[5] * Pv2
        + p[6] * Pw2
        + p[7] * Pu1 * Pv1
        + p[8] * Pu1 * Pw1
        + p[9] * Pu3
    )
    
    return result
# EVOLVE-BLOCK-END
