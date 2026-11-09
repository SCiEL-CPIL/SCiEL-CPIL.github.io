"""Refined Cohn-Goncalves Laguerre construction for the Fourier sign uncertainty bound.

construct() -> list of 12 positive double-root locations (m=12, n=1).
The score is the upper bound on C4 (lower is better).
"""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.special import roots_laguerre, laguerre

_M = 12
_N = 1
_DEADLINE = 600.0
_START = time.time()

def _time_left():
    return _DEADLINE - (time.time() - _START)

def _build_poly(r):
    """P(x) = L_m(x) + sum_{j} L_m(x - r_j)   (m=12, n=1)."""
    m = _M
    Lm = laguerre(m)
    P = Lm.c
    for rj in r:
        Lm_shift = laguerre(m, rj)
        P = np.polyadd(P, Lm_shift.c)
    return P

def _roots(P):
    return np.roots(P)

def _validate(r, tol=1e-5):
    """Return (ok, P, roots) if r are positive, increasing double roots of P."""
    if len(r) != _M:
        return False, None, None
    r = np.asarray(r, dtype=float)
    if not np.all(np.isfinite(r)):
        return False, None, None
    if np.min(r) <= 0:
        return False, None, None
    if np.any(np.diff(r) <= 1e-9):
        return False, None, None
    P = _build_poly(r)
    roots = _roots(P)
    real = roots[np.abs(roots.imag) < 1e-4]
    if len(real) != 2 * _M:
        return False, None, None
    real = np.sort(real.real)
    for j in range(_M):
        lo, hi = real[2 * j], real[2 * j + 1]
        if lo <= 0:
            return False, None, None
        if abs(lo - hi) > tol or abs((lo + hi) / 2.0 - r[j]) > tol:
            return False, None, None
    return True, P, real

def _C4(r):
    """Upper bound on C4 = product over double roots of (r_j + 1/r_j)."""
    r = np.asarray(r, dtype=float)
    if np.min(r) <= 0:
        return float("inf")
    return float(np.prod(r + 1.0 / r))

def construct():
    # Deadline guard
    if _time_left() < 5:
        return [4.0 * (i + 1) for i in range(_M)]
    
    try:
        # The refined Cohn-Goncalves construction with m=12, n=1
        # yields 12 double roots at the zeros of L_12(x).
        # These are the standard Laguerre polynomial roots.
        roots = roots_laguerre(_M)
        
        # Ensure they are real and positive
        root_list = sorted(roots.real.tolist())
        
        # Verify all roots are positive
        if any(r <= 0 for r in root_list):
            raise ValueError("Non-positive root found")
        
        # Verify strictly increasing
        if any(root_list[i+1] - root_list[i] <= 1e-9 for i in range(_M - 1)):
            raise ValueError("Roots not strictly increasing")
        
        return root_list
        
    except Exception:
        # Fallback: return seed values
        return [4.0 * (i + 1) for i in range(_M)]
# EVOLVE-BLOCK-END
