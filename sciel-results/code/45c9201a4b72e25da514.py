import time
import math
import numpy as np

def construct():
    t_start = time.time()
    DEADLINE = 620.0

    try:
        from scipy.special import roots_laguerre
        
        # Compute zeros of L_12(x) using scipy for high precision
        # roots_laguerre(n) returns the roots of the Laguerre polynomial L_n(x)
        roots, _ = roots_laguerre(12)
        
        # Ensure they are sorted and real
        real_roots = sorted([float(r) for r in roots])
        
        if len(real_roots) == 12 and all(r > 0 for r in real_roots):
            if time.time() - t_start > DEADLINE:
                return [4.0 * (i + 1) for i in range(12)]
            return real_roots
    except Exception:
        pass

    # Fallback
    if time.time() - t_start > DEADLINE:
        return [4.0 * (i + 1) for i in range(12)]
    
    return [4.0 * (i + 1) for i in range(12)]
