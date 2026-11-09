"""Rewritten program: use exact zeros of generalized Laguerre polynomial L_12^{(1)}(x)."""
import time
import math
import numpy as np

_START_TIME = time.time()
_HARD_DEADLINE = 560.0
M = 12
N = 1
NROOTS = 12

def _deadline_reached():
    return (time.time() - _START_TIME) >= _HARD_DEADLINE

def _compute_bound(r):
    """
    Compute the uncertainty bound for given double-root locations.
    
    For the Cohn-Goncalves construction with m=12, n=1, we have a polynomial
    P(x) = prod_{i=1}^{12} (x - r_i)^2.
    
    The uncertainty bound C4 involves integrals of the form:
    C4 = [integral_0^inf x^4 e^{-x} |P(x)|^2 dx] / [integral_0^inf e^{-x} |P(x)|^2 dx]
    """
    r = np.asarray(r, dtype=float)
    
    # Use Gauss-Laguerre quadrature for integrals with e^{-x} weight
    # Need enough points to resolve the polynomial of degree 24
    x_nodes, x_weights = np.polynomial.laguerre.laggauss(50)
    
    # Evaluate P(x) at each node
    P_vals = np.ones_like(x_nodes)
    for ri in r:
        P_vals *= (x_nodes - ri) ** 2
    
    # Compute integrals
    P2 = P_vals ** 2
    
    num = np.sum(x_weights * x_nodes**4 * P2)
    den = np.sum(x_weights * P2)
    
    if den < 1e-30:
        return 1e10  # very large penalty
    
    return num / den

def _laguerre_zeros(m, alpha):
    """
    Compute the zeros of the generalized Laguerre polynomial L_m^{(alpha)}(x).
    These are positive and well-separated, and should give a good uncertainty bound.
    """
    # Build the companion matrix for the monic generalized Laguerre polynomial
    # L_m^{(alpha)}(x) = sum_{k=0}^{m} (-1)^k * binom(m+alpha, m-k) * x^k / k!
    
    # Coefficients of L_m^{(alpha)}(x) in standard form (highest degree first)
    coeffs = np.zeros(m + 1)
    for k in range(m + 1):
        # Coefficient of x^k in L_m^{(alpha)}(x)
        # L_m^{(alpha)}(x) = sum_{k=0}^{m} (-1)^k * C(m+alpha, m-k) * x^k / k!
        # where C(a, b) = Gamma(a+1)/(Gamma(b+1)*Gamma(a-b+1))
        if k == 0:
            coeff = 1.0
        else:
            coeff = 0.0
            # Use the formula: L_m^{(alpha)}(x) = sum_{k=0}^{m} (-1)^k * binom(m+alpha, m-k) * x^k / k!
            # But we need to be careful with the sign and indexing
            # Let's use scipy's laguerre polynomial evaluation instead
    
    # Better approach: use scipy.special.laguerre to get the polynomial coefficients
    from scipy.special import roots_genlaguerre
    
    # roots_genlaguerre(n, alpha) returns the roots of L_n^{(alpha)}(x)
    zeros = roots_genlaguerre(m, alpha)
    
    return np.sort(zeros.real)

def _construct_core():
    """
    Use the exact zeros of the generalized Laguerre polynomial L_12^{(1)}(x)
    as the root locations.
    """
    # Get the zeros of L_12^{(1)}(x)
    r = _laguerre_zeros(M, N)
    
    # Ensure they are positive and sorted
    r = np.sort(r)
    r = np.maximum(r, 0.1)  # Ensure positivity
    
    # Verify we have exactly 12 positive roots
    if len(r) != NROOTS:
        # Fallback to seed configuration
        r = np.array([4.0 * (i + 1) for i in range(NROOTS)], dtype=float)
    
    return r.tolist()

def construct():
    """Return 12 positive double-root locations for the Laguerre combination
    (m=12, n=1).  Score is the upper bound on C4 (lower is better)."""
    try:
        result = _construct_core()
    except Exception:
        result = [4.0 * (i + 1) for i in range(NROOTS)]
    
    # Hard safety: if anything ran too long, return the known-good seed roots.
    if (time.time() - _START_TIME) >= _HARD_DEADLINE:
        result = [4.0 * (i + 1) for i in range(NROOTS)]
    
    # Final sanity: exactly 12 positive, increasing floats.
    if not (isinstance(result, list) and len(result) == NROOTS):
        result = [4.0 * (i + 1) for i in range(NROOTS)]
    out = []
    last = 0.0
    for v in result:
        v = float(v)
        if not (v > 0.0 and math.isfinite(v)):
            v = last + 1.0
        if v <= last:
            v = last + 1e-6
        out.append(v)
        last = v
    return out
