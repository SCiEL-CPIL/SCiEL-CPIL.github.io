"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np

# EVOLVE-BLOCK-START
_START_TIME = time.time()
_HARD_DEADLINE = 560.0   # safety margin before the 630 s wall-clock kill
M = 12
N = 1
NROOTS = 12

def _deadline_reached():
    return (time.time() - _START_TIME) >= _HARD_DEADLINE

def _get_laguerre_zeros(degree):
    """
    Compute the zeros of the standard Laguerre polynomial L_degree(x).
    These are the natural roots for Laguerre-based constructions.
    """
    try:
        from scipy.special import roots_laguerre
        # roots_laguerre returns (x, w) where x are the zeros
        roots, _ = roots_laguerre(degree)
        # Ensure they are real and positive
        roots = np.real(roots)
        # Sort them just in case, though they are usually sorted
        roots = np.sort(roots)
        return roots
    except ImportError:
        # Fallback if scipy is not available (though task says it is)
        # Use a simple approximation or seed
        return np.array([4.0 * (i + 1) for i in range(degree)])

def construct():
    """
    Return 12 positive double-root locations for the Laguerre combination
    (m=12, n=1). Score is the upper bound on C4 (lower is better).
    
    Strategy: Use the zeros of the Laguerre polynomial L_12(x).
    The seed uses arithmetic spacing (4, 8, ..., 48).
    The zeros of L_12 are clustered near 0 and spread out, which is the
    correct asymptotic behavior for Laguerre nodes. This structural change
    from uniform to orthogonal-polynomial-based nodes is expected to yield
    a tighter bound.
    """
    # Check deadline
    if _deadline_reached():
        return [4.0 * (i + 1) for i in range(NROOTS)]

    try:
        # Get the zeros of L_12
        roots = _get_laguerre_zeros(M)
        
        # Verify we have exactly 12 positive, finite, increasing roots
        if len(roots) != NROOTS:
            raise ValueError(f"Expected {NROOTS} roots, got {len(roots)}")
            
        # Filter and check positivity
        valid_roots = []
        last_val = 0.0
        for r in roots:
            r = float(r)
            if not math.isfinite(r) or r <= last_val:
                # If any issue, fall back to seed logic for that point or abort
                # But Laguerre roots are well-behaved.
                # For safety, if one is bad, we might need to adjust, 
                # but typically they are distinct and positive.
                pass
            
            # Ensure strictly increasing for the interface sanity check
            if r <= last_val:
                r = last_val + 1e-6
            
            valid_roots.append(r)
            last_val = r
            
        if len(valid_roots) != NROOTS:
             raise ValueError("Failed to construct 12 valid roots")
             
        return valid_roots

    except Exception as e:
        # Fallback to seed if any error occurs
        return [4.0 * (i + 1) for i in range(NROOTS)]

# EVOLVE-BLOCK-END
