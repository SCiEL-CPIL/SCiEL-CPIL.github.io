import numpy as np
import time

def _shifted_legendre(t, deg):
    """
    Compute shifted Legendre polynomials P_0(t) through P_deg(t) for t in [0, 1].
    Uses the standard three-term recurrence for numerical stability.
    
    Parameters
    ----------
    t : ndarray
        Values in [0, 1].
    deg : int
        Maximum degree of the polynomial.
    
    Returns
    -------
    ndarray
        Array of shape (deg + 1, N) where N is the size of t.
    """
    N = len(t)
    p = np.empty((deg + 1, N))
    p[0] = 1.0
    if deg >= 1:
        p[1] = 2.0 * t - 1.0
        for k in range(2, deg + 1):
            # Recurrence: P_k(t) = ((2k-1)(2t-1)P_{k-1}(t) - (k-1)P_{k-2}(t)) / k
            # This ensures standard normalization where P_k(1) = 1 and orthogonality
            # with weight 1 on [0,1].
            p[k] = ((2.0 * k - 1.0) * (2.0 * t - 1.0) * p[k-1] - (k - 1.0) * p[k-2]) / k
    return p

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid Basis: Shifted Legendre Polynomials (degree up to 3) with Log-Interactions.
    
    This model combines the strengths of two approaches:
    1. Shifted Legendre polynomials P0-P3 for each individual input to capture
       non-linear curvature in a well-conditioned, orthogonal manner.
       We include P3 terms to capture higher-order effects that quadratic models miss.
    2. Logarithmic cross-interaction terms to capture multiplicative dependencies
       (power-law behavior) which pure polynomials approximate poorly.
    
    Basis (10 terms):
    0: P0(x)P0(y)P0(z) (Constant 1)
    1: P1(x)P0(y)P0(z) (P1(x))
    2: P0(x)P1(y)P0(z) (P1(y))
    3: P0(x)P0(y)P1(z) (P1(z))
    4: P2(x)P0(y)P0(z) (P2(x))
    5: P0(x)P2(y)P0(z) (P2(y))
    6: P0(x)P0(y)P2(z) (P2(z))
    7: ln(x) * ln(y)
    8: ln(x) * ln(z)
    9: ln(y) * ln(z)
    
    Note: We sacrifice the P3 terms in favor of the log-interaction terms. 
    The log-interactions are often more critical for capturing the joint 
    behavior of positive inputs in physical or mathematical models, whereas 
    the P2 terms already capture significant curvature. The combination of 
    P2 and log-interactions is a stronger general prior for symbolic regression 
    on positive domains than pure polynomials or pure log-polynomials.
    """
    p = params
    
    # Normalize inputs from [0.5, 2] to [0, 1] for Legendre basis
    # This ensures the basis functions are well-scaled and orthogonal on [0,1]
    tx = (x - 0.5) / 1.5
    ty = (y - 0.5) / 1.5
    tz = (z - 0.5) / 1.5
    
    # Compute Legendre basis for each normalized input
    # We need up to degree 2 for the basis terms used below
    # We compute up to degree 3 to have flexibility, but only use up to P2
    x0, x1, x2, _ = _shifted_legendre(tx, 3)
    y0, y1, y2, _ = _shifted_legendre(ty, 3)
    z0, z1, z2, _ = _shifted_legendre(tz, 3)
    
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
    return (p[0] * x0 * y0 * z0 + 
            p[1] * x1 * y0 * z0 + 
            p[2] * x0 * y1 * z0 + 
            p[3] * x0 * y0 * z1 + 
            p[4] * x2 * y0 * z0 + 
            p[5] * x0 * y2 * z0 + 
            p[6] * x0 * y0 * z2 + 
            p[7] * lxy + 
            p[8] * lxz + 
            p[9] * lyz)
