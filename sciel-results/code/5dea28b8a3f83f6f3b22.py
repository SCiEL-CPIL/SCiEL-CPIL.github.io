import time
import numpy as np
from numpy.polynomial import laguerre

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 600  # seconds
    
    try:
        # Compute the roots of the Laguerre polynomial L_12(x)
        # These are the optimal nodes for the specified construction
        nodes, _ = laguerre.laggauss(12)
        
        # Convert to list of floats
        result = [float(x) for x in nodes]
        
        # Check deadline (though this is very fast)
        if time.time() - start_time > deadline - 10:
            return result
            
        return result
        
    except Exception:
        # Fallback to a reasonable approximation if numpy fails (unlikely)
        # Using the asymptotic formula for the largest root is not enough for all 12
        # But we can return the seed-like values as a last resort
        return [4.0 * (i + 1) for i in range(12)]
# EVOLVE-BLOCK-END
