"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import roots_laguerre

def construct():
    """Return 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    
    Uses the zeros of the Laguerre polynomial L_12(x) as the natural double-root
    locations for the Cohn-Goncalves construction.
    """
    # Get the 12 zeros of L_12(x)
    roots, weights = roots_laguerre(12)
    # Ensure positive (they should all be positive for Laguerre)
    return [float(r) for r in sorted(roots)]
# EVOLVE-BLOCK-END
