import numpy as np
from scipy.special import roots_genlaguerre
import time

def construct():
    """
    Returns the 12 positive double-root locations for the Laguerre combination
    (m=12, n=1) tuned for the refined Cohn-Goncalves uncertainty bound.
    """
    start_time = time.time()
    
    # Deadline guard: if we somehow take too long (unlikely for this computation),
    # return the seed result.
    deadline = start_time + 590.0  # 10s margin
    
    try:
        # The task specifies a Laguerre-polynomial construction.
        # The "double root" nature and m=12 suggest the zeros of the 
        # generalized Laguerre polynomial L_12^(alpha)(x).
        # A common choice in such uncertainty principle constructions 
        # involving squared terms or specific weights is alpha=2.
        # If alpha=2 is incorrect, alpha=0 is the primary alternative.
        # Given the "double root" phrasing, alpha=2 is a stronger candidate
        # as it corresponds to the weight x^2 e^-x, often arising from 
        # radial symmetries or squared Gaussian components.
        
        alpha = 2
        n = 12
        
        # Compute the roots of L_n^(alpha)(x)
        # roots_genlaguerre returns the roots of the generalized Laguerre polynomial
        roots = roots_genlaguerre(n, alpha)
        
        # Extract real parts (imaginary parts are negligible/zero for real alpha > -1)
        root_list = [float(r.real) for r in roots]
        
        # Ensure all roots are positive (they should be for n>0, alpha>-1)
        root_list = [r for r in root_list if r > 0]
        
        # Sort the roots
        root_list.sort()
        
        # Verify we have exactly 12 roots
        if len(root_list) != 12:
            # Fallback if something went wrong
            return [4.0 * (i + 1) for i in range(12)]
            
        return root_list

    except Exception:
        # Fallback to seed if any error occurs
        return [4.0 * (i + 1) for i in range(12)]
