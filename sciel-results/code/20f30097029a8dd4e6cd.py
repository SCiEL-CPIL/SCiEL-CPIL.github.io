import numpy as np
from scipy.optimize import minimize
from scipy.linalg import eigh_tridiagonal
from scipy.special import roots_laguerre

def construct():
    """
    Returns the 12 positive double-root locations corresponding to the
    standard Laguerre polynomial L_12. The standard Laguerre roots are
    known to minimize the L2 energy of the monic polynomial with respect
    to the Laguerre weight, which serves as the optimal configuration for
    the refined uncertainty bound.
    """
    N = 12
    # Fixed diagonal entries based on Laguerre recurrence
    d = np.array([2*i + 1 for i in range(N)], dtype=float)
    # Standard off-diagonal entries for Laguerre L_12
    o_std = np.array([i for i in range(1, N)], dtype=float)
    
    # Compute eigenvalues of the standard Jacobi matrix
    # This is equivalent to roots_laguerre(N)
    w, _ = eigh_tridiagonal(d, o_std, select='all')
    
    # Ensure positive and distinct (eigh_tridiagonal guarantees distinct if o != 0)
    if np.any(w <= 0):
        # Fallback to direct calculation if numerical issues arise
        roots, _ = roots_laguerre(12)
        return [float(r) for r in roots]
        
    return [float(r) for r in w]
