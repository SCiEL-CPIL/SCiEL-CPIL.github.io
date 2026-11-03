"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import roots_laguerre

def construct():
    """Return 12 positive double-root locations for the Laguerre combination (m=12, n=1)."""
    # Use zeros of the Laguerre polynomial L_12^{(2)}(x) for n=1 (alpha=2n=2)
    # These are the natural roots for the Cohn-Goncalves construction
    try:
        from scipy.special import roots_laguerre
        # roots_laguerre gives roots of L_n(x), i.e., L_n^{(0)}(x)
        # For the associated Laguerre L_m^{(alpha)}, we may need a different approach
        # Let's use the standard Laguerre roots L_12^{(0)} first
        r, _ = roots_laguerre(12)
        # Ensure positive and sorted
        r = sorted(r)
        # Verify all positive
        if all(x > 0 for x in r):
            return [float(x) for x in r]
    except Exception:
        pass
    
    # Fallback: equally spaced (same as seed)
    return [4.0 * (i + 1) for i in range(12)]
# EVOLVE-BLOCK-END
