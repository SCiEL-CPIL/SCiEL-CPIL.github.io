"""Model-and-optimize approach for symbolic regression with 3 inputs."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Basis of shifted Legendre polynomials of degree up to 3 on normalized inputs,
    plus one cross-interaction term.
    
    Parameters (10 total):
    0: constant
    1: P1(u)
    2: P2(u)
    3: P3(u)
    4: P1(v)
    5: P2(v)
    6: P3(v)
    7: P1(w)
    8: P2(w)
    9: P1(u) * P1(v)
    """
    p = params
    
    # Normalize inputs to [-1, 1]
    # x, y, z are in [0.5, 2.0]
    # center = 1.25, scale = 0.75
    u = (x - 1.25) / 0.75
    v = (y - 1.25) / 0.75
    w = (z - 1.25) / 0.75
    
    # Shifted Legendre polynomials on [-1, 1]
    # P0(t) = 1
    # P1(t) = t
    # P2(t) = (3*t^2 - 1) / 2
    # P3(t) = (5*t^3 - 3*t) / 2
    
    P1_u = u
    P2_u = (3 * u**2 - 1) / 2
    P3_u = (5 * u**3 - 3 * u) / 2
    
    P1_v = v
    P2_v = (3 * v**2 - 1) / 2
    P3_v = (5 * v**3 - 3 * v) / 2
    
    P1_w = w
    P2_w = (3 * w**2 - 1) / 2
    P3_w = (5 * w**3 - 3 * w) / 2
    
    return (p[0] +
            p[1] * P1_u +
            p[2] * P2_u +
            p[3] * P3_u +
            p[4] * P1_v +
            p[5] * P2_v +
            p[6] * P3_v +
            p[7] * P1_w +
            p[8] * P2_w +
            p[9] * P1_u * P1_v)
# EVOLVE-BLOCK-END
