"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import roots_genlaguerre

def construct():
    """Return 12 positive double-root locations for the Laguerre combination (m=12, n=1)."""
    # Get the roots of the generalized Laguerre polynomial L_12^(1)(x)
    # These are the natural roots for the Cohn-Gonçalves construction
    roots = roots_genlaguerre(12, 1)
    # Ensure they are positive and sorted
    roots = sorted(roots.real)
    # Return as a list of floats
    return [float(r) for r in roots]
# EVOLVE-BLOCK-END
