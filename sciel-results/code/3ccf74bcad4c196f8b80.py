"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Shifted Legendre polynomial basis (degree 0-3) with cross-interaction terms.
    Normalizes inputs from [0.5, 2.0] to [-1, 1] for orthogonal, well-conditioned basis.
    
    Basis:
      P0(t) = 1
      P1(t) = t
      P2(t) = 0.5*(3t^2 - 1)
      P3(t) = 0.5*(5t^3 - 3t)
    
    Terms (10 total):
      p[0]:  P0(u)  (constant)
      p[1]:  P1(u)  (linear in x)
      p[2]:  P1(v)  (linear in y)
      p[3]:  P1(w)  (linear in z)
      p[4]:  P2(u)  (quadratic in x)
      p[5]:  P2(v)  (quadratic in y)
      p[6]:  P2(w)  (quadratic in z)
      p[7]:  P1(u)*P1(v)  (interaction x*y)
      p[8]:  P1(u)*P1(w)  (interaction x*z)
      p[9]:  P1(v)*P1(w)  (interaction y*z)
    """
    p = params
    
    # Normalize inputs to [-1, 1]
    u = (x - 1.25) / 0.75
    v = (y - 1.25) / 0.75
    w = (z - 1.25) / 0.75
    
    # Shifted Legendre polynomials
    P1_u = u
    P1_v = v
    P1_w = w
    
    P2_u = 0.5 * (3.0 * u * u - 1.0)
    P2_v = 0.5 * (3.0 * v * v - 1.0)
    P2_w = 0.5 * (3.0 * w * w - 1.0)
    
    # Cross-interaction terms (linear-linear)
    uv = P1_u * P1_v
    uw = P1_u * P1_w
    vw = P1_v * P1_w
    
    return (p[0] +
            p[1] * P1_u +
            p[2] * P1_v +
            p[3] * P1_w +
            p[4] * P2_u +
            p[5] * P2_v +
            p[6] * P2_w +
            p[7] * uv +
            p[8] * uw +
            p[9] * vw)
# EVOLVE-BLOCK-END
