"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import numpy.polynomial.laguerre as npl
import time

def construct():
    start_time = time.time()
    deadline = 600.0
    safety_margin = 30.0  # seconds before the 630s kill signal
    
    # The task asks for 12 positive double-root locations for a Laguerre combination
    # with m=12 and n=1. The "double roots" phrasing in the context of Cohn-Goncalves
    # often refers to the nodes of the Gauss-Laguerre quadrature, which are the
    # simple roots of the Laguerre polynomial L_m(x). The seed program used a
    # simple arithmetic progression [4, 8, ..., 48]. The inspirations suggest
    # using the actual roots of L_12(x).
    
    # Let's try to get the roots of L_12(x) using numpy's laggauss.
    # laggauss(n) returns the nodes (roots) and weights for Gauss-Laguerre quadrature.
    # The nodes are the roots of L_n(x).
    
    try:
        # Get the roots of L_12(x)
        roots, weights = npl.laggauss(12)
        
        # Ensure they are positive and sorted
        roots = np.sort(roots).astype(float)
        
        # Check if we have 12 positive roots
        if len(roots) == 12 and np.all(roots > 0):
            return roots.tolist()
        else:
            # Fallback to seed if something is wrong
            return [4.0 * (i + 1) for i in range(12)]
            
    except Exception:
        # Fallback to seed if an exception occurs
        return [4.0 * (i + 1) for i in range(12)]
# EVOLVE-BLOCK-END
