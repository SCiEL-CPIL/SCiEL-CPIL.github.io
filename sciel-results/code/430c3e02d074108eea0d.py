"""Refined Cohn-Goncalves Laguerre-polynomial construction for the Fourier
sign uncertainty upper bound.

This version explores generalized Laguerre polynomials L_n^{(alpha)} with 
different alpha values. The standard construction uses alpha=0. 
Hypothesis: Shifting roots towards the origin (using alpha < 0) may reduce 
the upper bound on C4, assuming the bound is monotonically related to 
the spread or magnitude of the roots.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.special import roots_genlaguerre, roots_laguerre

_START_TIME = time.time()
_DEADLINE = 630.0          # evaluator kills at 630 s wall-clock
_SAFETY = 30.0             # return well before the hard limit
_M = 12
_N = 1


def _deadline_exceeded():
    return (time.time() - _START_TIME) > (_DEADLINE - _SAFETY)


def _get_roots_for_alpha(alpha):
    """Get roots of L_M^{(alpha)}."""
    try:
        if alpha == 0:
            zeros, _ = roots_laguerre(_M)
        else:
            zeros, _ = roots_genlaguerre(_M, alpha)
        zeros = np.sort(zeros)
        # Ensure positivity
        if not np.all(zeros > 0):
            return None
        return [float(z) for z in zeros]
    except Exception:
        return None


def construct():
    """Return a list of 12 positive double-root locations for the Laguerre
    combination (m=12, n=1). Score (upper bound on C4) is the raw value of
    the resulting construction; lower is better.
    """
    best_roots = None
    best_score_proxy = float('inf')
    
    # Try different alpha values.
    # alpha=0 is the standard.
    # alpha < 0 shifts roots to the left (smaller values).
    # If the bound is proportional to sum of roots or max root, smaller is better.
    alphas = [0.0, -0.5, -0.9, 0.5, 1.0]
    
    for alpha in alphas:
        if _deadline_exceeded():
            break
        roots = _get_roots_for_alpha(alpha)
        if roots is None:
            continue
        
        # Proxy score: Sum of roots. 
        # Rationale: If the uncertainty bound is related to the "size" of the 
        # support or the polynomial's growth, smaller roots might be better.
        # This is a heuristic structural change.
        proxy = sum(roots)
        
        # Also consider the product of roots (discriminant related)
        # But sum is a simpler first-order metric.
        
        if proxy < best_score_proxy:
            best_score_proxy = proxy
            best_roots = roots

    if best_roots is None:
        # Fallback to standard L_12
        try:
            zeros, _ = roots_laguerre(_M)
            best_roots = [float(z) for z in zeros]
        except:
            best_roots = [4.0 * (i + 1) for i in range(_M)]
            
    return best_roots
# EVOLVE-BLOCK-END
