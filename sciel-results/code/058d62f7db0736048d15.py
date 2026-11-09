import numpy as np
from scipy.special import roots_laguerre
import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = 600.0
    
    # The task specifies m=12, n=1.
    # The seed uses a uniform grid 4, 8, ..., 48.
    # Inspiring solutions using L_12 (alpha=0) were worse.
    # This suggests the optimal roots are shifted to the right.
    # The mean of the grid is 26.
    # The mean of the roots of L_m(x; alpha) is m + alpha.
    # To match the grid mean, we need 12 + alpha = 26 => alpha = 14.
    # We use alpha=14 as a strong candidate for the "refined" construction.
    
    m = 12
    alpha = 14.0
    
    try:
        # roots_laguerre returns (roots, weights)
        roots, weights = roots_laguerre(m, alpha)
        # Ensure roots are positive and sorted
        roots = sorted(roots)
        
        # Check time
        if time.time() - start_time > deadline - 10:
            return [4.0 * (i + 1) for i in range(12)]
            
        return roots
        
    except Exception:
        # Fallback to seed
        return [4.0 * (i + 1) for i in range(12)]
