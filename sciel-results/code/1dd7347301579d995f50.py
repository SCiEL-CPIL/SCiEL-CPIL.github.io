"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import roots_genlaguerre

def construct():
    """Return 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    
    The roots of the generalized Laguerre polynomial L_12^(1)(x) are the natural
    double-root locations in the Cohn-Goncalves construction. The score (upper bound
    on C4) is determined by the sign-change structure, with the last sign change
    at the largest root.
    """
    m = 12
    n = 1
    # roots_genlaguerre(m, n) returns the m roots of L_m^(n)(x)
    roots = roots_genlaguerre(m, n)
    # Sort in increasing order (they should already be positive and sorted)
    roots = np.sort(roots)
    return roots.tolist()
# EVOLVE-BLOCK-END

