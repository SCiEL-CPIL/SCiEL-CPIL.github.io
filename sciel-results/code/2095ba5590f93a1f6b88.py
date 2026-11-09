"""Direct construction using the Cohn-Goncalves Laguerre polynomial root structure."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from numpy.polynomial.laguerre import laggauss
from scipy.special import genlaguerre, eval_genlaguerre

_START_TIME = time.time()
_DEADLINE = 600.0
_SAFETY_MARGIN = 30.0

def _deadline_reached():
    return (time.time() - _START_TIME) >= (_DEADLINE - _SAFETY_MARGIN)

def _compute_laguerre_roots(m, alpha):
    """Compute the m positive roots of the generalized Laguerre polynomial L_m^{(alpha)}(x)."""
    # Use numpy's laggauss for standard Laguerre (alpha=0)
    if alpha == 0:
        roots, _ = laggauss(m)
        return roots
    # For general alpha, use scipy's genlaguerre to get coefficients, then find roots
    # genlaguerre(m, alpha) returns a poly1d-like object
    coeffs = genlaguerre(m, alpha)
    # coeffs is a poly1d with coefficients highest degree first
    roots = np.roots(coeffs)
    # Filter to positive real roots
    roots_real = roots[np.abs(roots.imag) < 1e-10].real
    roots_positive = roots_real[roots_real > 1e-10]
    return np.sort(roots_positive)

def _fallback_construct():
    """Fallback: use standard Laguerre roots (alpha=0) if general case fails."""
    roots, _ = laggauss(12)
    return [float(r) for r in roots]

def construct():
    m = 12
    n = 1
    
    # In the Cohn-Goncalves construction for Fourier sign uncertainty,
    # the relevant Laguerre polynomial is L_m^{(2n-1)} or L_m^{(2n)}.
    # Try alpha = 2n - 1 = 1 first, then alpha = 2n = 2.
    # We want to minimize the largest root (score = 4 * max_root).
    
    candidates = []
    
    for alpha in [0, 1, 2]:
        try:
            roots = _compute_laguerre_roots(m, alpha)
            if len(roots) == m:
                candidates.append((float(np.max(roots)), alpha, roots))
        except Exception:
            continue
    
    if candidates:
        # Pick the one with smallest maximum root
        candidates.sort(key=lambda x: x[0])
        best_roots = candidates[0][2]
        return [float(r) for r in best_roots]
    else:
        return _fallback_construct()
# EVOLVE-BLOCK-END
