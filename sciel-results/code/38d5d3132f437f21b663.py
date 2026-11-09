"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.special import roots_genlaguerre
import time

# EVOLVE-BLOCK-START
def construct():
    """
    Construct 12 positive double-root locations for the Laguerre combination
    (m=12, n=1) tuned for the Fourier sign uncertainty upper bound.
    
    Uses zeros of the generalized Laguerre polynomial L_12^(alpha) as a starting
    point, with alpha chosen to match the dimension n=1 context.
    """
    # Try using zeros of L_12^(0) (standard Laguerre) scaled by a factor
    # The standard Laguerre L_12 zeros are in (0, ~25)
    # We need to figure out the right scaling
    
    # Let's try several candidates and return the one that seems most "optimal"
    # based on known properties of the Cohn-Goncalves construction
    
    # Candidate 1: Zeros of L_12^(0) scaled by 4 (same total range as seed)
    try:
        roots_lag0 = roots_genlaguerre(12, 0.0)[0]
        # Scale so that max root ~ 48 (like seed)
        scale1 = 48.0 / max(roots_lag0)
        candidate1 = list(roots_lag0 * scale1)
    except:
        candidate1 = [4.0 * (i + 1) for i in range(12)]
    
    # Candidate 2: Zeros of L_12^(1) (alpha=1, related to n=1)
    try:
        roots_lag1 = roots_genlaguerre(12, 1.0)[0]
        scale2 = 48.0 / max(roots_lag1)
        candidate2 = list(roots_lag1 * scale2)
    except:
        candidate2 = candidate1
    
    # Candidate 3: Power-law spacing r_i = c * (i+1)^p
    # Choose p and c to span similar range
    p = 1.5
    c = 48.0 / (12 ** p)
    candidate3 = list(c * ((i + 1) ** p) for i in range(12))
    
    # Candidate 4: Geometric spacing
    ratio = (48.0 / 4.0) ** (1.0 / 11.0)
    candidate4 = list(4.0 * (ratio ** i) for i in range(12))
    
    # Return candidate1 (Laguerre zeros) as it's most theoretically grounded
    return candidate1
# EVOLVE-BLOCK-END
