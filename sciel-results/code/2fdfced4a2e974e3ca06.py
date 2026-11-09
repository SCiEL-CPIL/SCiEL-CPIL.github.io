import numpy as np

def _legendre_basis(t):
    """
    Compute shifted Legendre polynomials P0..P3 on t in [0,1].
    
    P0(t) = 1
    P1(t) = 2t - 1
    P2(t) = 6t^2 - 6t + 1
    P3(t) = 20t^3 - 30t^2 + 12t - 1
    """
    p0 = np.ones_like(t)
    p1 = 2.0 * t - 1.0
    p2 = 6.0 * t * t - 6.0 * t + 1.0
    p3 = 20.0 * t * t * t - 30.0 * t * t + 12.0 * t - 1.0
    return p0, p1, p2, p3

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Shifted Legendre Polynomial Basis with Cross-Interactions.
    
    This model uses orthogonal Legendre polynomials of degree up to 3 applied
    to normalized inputs (mapped from [0.5, 2] to [0, 1]). Orthogonal bases
    provide better numerical conditioning for BFGS fitting and capture higher-
    order curvature more efficiently than raw monomials.
    
    Basis (10 terms):
    0: P0 = 1 (constant)
    1: P1(x)
    2: P1(y)
    3: P1(z)
    4: P2(x)
    5: P2(y)
    6: P2(z)
    7: P1(x) * P1(y)
    8: P1(x) * P1(z)
    9: P1(y) * P1(z)
    """
    p = params
    
    # Normalize inputs from [0.5, 2] to [0, 1]
    tx = (x - 0.5) / 1.5
    ty = (y - 0.5) / 1.5
    tz = (z - 0.5) / 1.5
    
    # Compute Legendre polynomial values
    _, p1x, p2x, _ = _legendre_basis(tx)
    _, p1y, p2y, _ = _legendre_basis(ty)
    _, p1z, p2z, _ = _legendre_basis(tz)
    
    # Cross-interaction terms
    p1xy = p1x * p1y
    p1xz = p1x * p1z
    p1yz = p1y * p1z
    
    # Linear combination
    return (p[0] + 
            p[1] * p1x + 
            p[2] * p1y + 
            p[3] * p1z + 
            p[4] * p2x + 
            p[5] * p2y + 
            p[6] * p2z + 
            p[7] * p1xy + 
            p[8] * p1xz + 
            p[9] * p1yz)
