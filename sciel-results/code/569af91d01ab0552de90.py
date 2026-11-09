"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 620.0  # safety margin before 630s kill

def construct():
    """
    Returns list of 12 positive double-root locations for the Laguerre
    combination (m=12, n=1) that minimizes the C4 upper bound.
    
    Strategy:
    Use the roots of the Generalized Laguerre polynomial L_12^(1)(x).
    This polynomial is orthogonal with respect to the weight x^1 * exp(-x),
    which matches the problem parameters (n=1).
    The roots of orthogonal polynomials typically minimize the weighted
    L2 norm of the polynomial, which is a proxy for the C4 bound.
    """
    # Deadline guard
    if time.time() - _START_TIME > _TIME_LIMIT:
        return [4.0 * (i + 1) for i in range(12)]

    try:
        from scipy.special import roots_genlaguerre
        
        # Compute roots of L_12^(1)(x)
        # alpha=1 corresponds to weight x^1 * exp(-x)
        roots, _ = roots_genlaguerre(12, 1)
        
        # Ensure roots are positive and sorted
        roots = np.sort(roots)
        
        # Verify positivity (Laguerre roots are positive)
        if np.any(roots <= 0):
            raise ValueError("Non-positive roots found")
            
        return [float(x) for x in roots]
        
    except Exception as e:
        # Fallback to seed if scipy fails or any other error
        # The seed is [4.0, 8.0, ..., 48.0]
        return [4.0 * (i + 1) for i in range(12)]

# EVOLVE-BLOCK-END
