import numpy as np
import time

def _legendre_basis(t):
    """
    Compute shifted Legendre polynomials P0..P3 on t in [0,1].
    """
    p0 = np.ones_like(t)
    p1 = 2.0 * t - 1.0
    p2 = 6.0 * t * t - 6.0 * t + 1.0
    p3 = 20.0 * t * t * t - 30.0 * t * t + 12.0 * t - 1.0
    return p0, p1, p2, p3

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid Basis: Shifted Legendre Polynomials (degree up to 3) with Log-Interactions.
    
    This model combines the strengths of two approaches:
    1. Shifted Legendre polynomials P0-P3 for each individual input to capture
       non-linear curvature in a well-conditioned, orthogonal manner.
    2. Logarithmic cross-interaction terms to capture multiplicative dependencies
       (power-law behavior) which pure polynomials approximate poorly.
    
    Basis (10 terms):
    0: 1 (constant)
    1: P1(x)
    2: P1(y)
    3: P1(z)
    4: P2(x)
    5: P2(y)
    6: P2(z)
    7: ln(x) * ln(y)
    8: ln(x) * ln(z)
    9: ln(y) * ln(z)
    
    Note: P3 terms are computed but not used in the linear combination to reserve
    parameters for the potentially more critical interaction terms. If the target
    function is highly non-linear in single variables, the P2 terms capture that,
    while the log-interactions capture the joint behavior.
    """
    p = params
    
    # Normalize inputs from [0.5, 2] to [0, 1] for Legendre basis
    tx = (x - 0.5) / 1.5
    ty = (y - 0.5) / 1.5
    tz = (z - 0.5) / 1.5
    
    # Compute Legendre polynomial values
    _, p1x, p2x, _ = _legendre_basis(tx)
    _, p1y, p2y, _ = _legendre_basis(ty)
    _, p1z, p2z, _ = _legendre_basis(tz)
    
    # Compute log values for interaction terms
    # Inputs are in [0.5, 2], so log is safe and real
    lx = np.log(x)
    ly = np.log(y)
    lz = np.log(z)
    
    # Compute log interaction terms
    lxy = lx * ly
    lxz = lx * lz
    lyz = ly * lz
    
    # Linear combination
    return (p[0] + 
            p[1] * p1x + 
            p[2] * p1y + 
            p[3] * p1z + 
            p[4] * p2x + 
            p[5] * p2y + 
            p[6] * p2z + 
            p[7] * lxy + 
            p[8] * lxz + 
            p[9] * lyz)
