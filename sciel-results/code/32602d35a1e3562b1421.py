import numpy as np

def _shifted_legendre(t, deg):
    """
    Compute shifted Legendre polynomials P_0(t) through P_deg(t) for t in [0, 1].
    Uses the three-term recurrence for numerical stability.
    
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
    p = np.empty((deg + 1, len(t)))
    p[0] = 1.0
    if deg >= 1:
        p[1] = 2.0 * t - 1.0
        for k in range(2, deg + 1):
            # Shifted Legendre recurrence:
            # P_k(x) = ((2k-1)(2x-1)P_{k-1}(x) - (k-1)P_{k-2}(x)) / k
            # Wait, the recurrence in the previous code was:
            # p[k] = (2.0 * k + 1.0) * (2.0 * t - 1.0) * p[k-1] - (k - 1.0) * p[k-2]
            # This is missing the division by k?
            # Standard recurrence: (k+1)P_{k+1}(x) = (2k+1)xP_k(x) - kP_{k-1}(x)
            # So P_k(x) = ((2k-1)xP_{k-1}(x) - (k-1)P_{k-2}(x)) / k
            # The previous code's recurrence was: p[k] = (2k+1)(2t-1)p[k-1] - (k-1)p[k-2]
            # This looks like it's computing unnormalized polynomials or using a different convention.
            # Let's use the standard normalized recurrence to be safe and ensure orthogonality.
            p[k] = ((2.0 * k - 1.0) * (2.0 * t - 1.0) * p[k-1] - (k - 1.0) * p[k-2]) / k
    return p

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic Regression using Shifted Legendre Polynomial Basis.
    
    This function uses a basis of shifted Legendre polynomials up to degree 2
    for each variable, plus pairwise interaction terms of degree 1x1.
    
    Basis Terms (10 total):
    0: 1 (Constant)
    1: P1(x)
    2: P1(y)
    3: P1(z)
    4: P2(x)
    5: P2(y)
    6: P2(z)
    7: P1(x) * P1(y)
    8: P1(x) * P1(z)
    9: P1(y) * P1(z)
    
    This basis is orthogonal (in the univariate sense) and well-conditioned,
    which helps the BFGS optimizer converge to the least-squares solution
    more reliably than raw monomials.
    """
    p = params
    
    # Normalize inputs from [0.5, 2] to [0, 1]
    xn = (x - 0.5) / 1.5
    yn = (y - 0.5) / 1.5
    zn = (z - 0.5) / 1.5
    
    # Compute Legendre polynomials up to degree 2
    # We only need P1 and P2 for the basis, but we compute P0, P1, P2
    lx = _shifted_legendre(xn, 2)
    ly = _shifted_legendre(yn, 2)
    lz = _shifted_legendre(zn, 2)
    
    # Extract specific terms
    # lx[0] is P0 (all 1s), lx[1] is P1, lx[2] is P2
    term0 = 1.0
    term1 = lx[1]
    term2 = ly[1]
    term3 = lz[1]
    term4 = lx[2]
    term5 = ly[2]
    term6 = lz[2]
    term7 = lx[1] * ly[1]
    term8 = lx[1] * lz[1]
    term9 = ly[1] * lz[1]
    
    # Construct prediction
    pred = (p[0] * term0 + 
            p[1] * term1 + 
            p[2] * term2 + 
            p[3] * term3 + 
            p[4] * term4 + 
            p[5] * term5 + 
            p[6] * term6 + 
            p[7] * term7 + 
            p[8] * term8 + 
            p[9] * term9)
            
    return pred
