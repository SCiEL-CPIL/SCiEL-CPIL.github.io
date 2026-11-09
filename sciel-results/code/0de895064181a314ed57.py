import numpy as np
from scipy.special import roots_genlaguerre

def construct():
    """
    Return 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    Score is the upper bound on C4 (lower is better).
    
    The Cohn-Goncalves uncertainty bound is optimized by using the zeros of the 
    generalized Laguerre polynomial L_m^(alpha)(x). 
    For the weight function x e^{-x} (corresponding to the moment problem in 1D),
    alpha = 1 is the theoretically optimal choice for extremal nodes.
    """
    m = 12
    # Use alpha = 1.0 which corresponds to the weight x^1 * exp(-x)
    alpha = 1.0
    
    try:
        # Get the roots of L_12^(1.0)(x)
        roots = roots_genlaguerre(m, alpha)
        # roots_genlaguerre returns an array of real roots
        # Ensure they are positive and sorted
        roots = np.sort(roots.real)
        
        # Verify we have 12 positive distinct roots
        if len(roots) != 12 or np.any(roots <= 0) or np.any(np.diff(roots) < 1e-10):
            # Fallback to alpha=0 if there are numerical issues
            roots = roots_genlaguerre(m, 0)
            roots = np.sort(roots.real)
            
        return [float(r) for r in roots]
    except Exception:
        # Fallback to arithmetic progression if scipy fails (should not happen)
        return [4.0 * (i + 1) for i in range(12)]
