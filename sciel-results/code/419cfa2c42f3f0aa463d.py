"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.special import roots_laguerre

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 590.0  # 10s safety margin
    
    # Use zeros of the Laguerre polynomial L_12^(1)(x) as double-root locations
    # n=1 suggests alpha=1 for the generalized Laguerre polynomial
    try:
        # Get roots of L_12^(1)(x)
        x, w = roots_laguerre(12, alpha=1)
        roots = sorted(x.tolist())
        
        # Verify constraints: positive and strictly increasing
        if len(roots) == 12 and all(r > 0 for r in roots) and all(
            roots[i+1] > roots[i] for i in range(11)
        ):
            return roots
    except Exception:
        pass
    
    # Fallback: try alpha=0
    try:
        x, w = roots_laguerre(12, alpha=0)
        roots = sorted(x.tolist())
        if len(roots) == 12 and all(r > 0 for r in roots):
            return roots
    except Exception:
        pass
    
    # Final fallback: seed values
    return [4.0 * (i + 1) for i in range(12)]
# EVOLVE-BLOCK-END
