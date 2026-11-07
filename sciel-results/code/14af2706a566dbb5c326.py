# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    f(x,y,z) using a basis of tensor-product Legendre polynomials.
    This basis spans the same space as the full quadratic polynomial
    (degree <= 2) but is approximately orthogonal over the domain [0.5, 2]^3.
    This improves the condition number of the design matrix, aiding BFGS convergence.
    
    Variables are mapped to [-1, 1]:
    u = (4x - 5) / 3
    v = (4y - 5) / 3
    w = (4z - 5) / 3
    
    Basis functions (10 total):
    params[0]: P0(u)P0(v)P0(w) = 1
    params[1]: P1(u)P0(v)P0(w) = u
    params[2]: P0(u)P1(v)P0(w) = v
    params[3]: P0(u)P0(v)P1(w) = w
    params[4]: P2(u)P0(v)P0(w) = 0.5 * (3*u^2 - 1)
    params[5]: P1(u)P1(v)P0(w) = u * v
    params[6]: P1(u)P0(v)P1(w) = u * w
    params[7]: P0(u)P2(v)P0(w) = 0.5 * (3*v^2 - 1)
    params[8]: P0(u)P1(v)P1(w) = v * w
    params[9]: P0(u)P0(v)P2(w) = 0.5 * (3*w^2 - 1)
    """
    c0, c1, c2, c3, c4, c5, c6, c7, c8, c9 = params
    
    # Map inputs to [-1, 1]
    u = (4.0 * x - 5.0) / 3.0
    v = (4.0 * y - 5.0) / 3.0
    w = (4.0 * z - 5.0) / 3.0
    
    # Compute Legendre polynomial components
    # P0(t) = 1
    # P1(t) = t
    # P2(t) = 0.5 * (3*t^2 - 1)
    
    p0_u = 1.0
    p1_u = u
    p2_u = 0.5 * (3.0 * u * u - 1.0)
    
    p0_v = 1.0
    p1_v = v
    p2_v = 0.5 * (3.0 * v * v - 1.0)
    
    p0_w = 1.0
    p1_w = w
    p2_w = 0.5 * (3.0 * w * w - 1.0)
    
    # Construct the linear combination
    return (c0 * p0_u * p0_v * p0_w +
            c1 * p1_u * p0_v * p0_w +
            c2 * p0_u * p1_v * p0_w +
            c3 * p0_u * p0_v * p1_w +
            c4 * p2_u * p0_v * p0_w +
            c5 * p1_u * p1_v * p0_w +
            c6 * p1_u * p0_v * p1_w +
            c7 * p0_u * p2_v * p0_w +
            c8 * p0_u * p1_v * p1_w +
            c9 * p0_u * p0_v * p2_w)
# EVOLVE-BLOCK-END
