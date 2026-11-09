"""Refined Cohn-Goncalves Laguerre construction for Fourier sign uncertainty."""
import time
import numpy as np
from scipy.special import roots_genlaguerre

# EVOLVE-BLOCK-START
def construct():
    """Return 12 positive double-root locations for the Laguerre combination (m=12, n=1)."""
    deadline = time.time() + 580
    
    best_roots = None
    best_product = -1
    
    # Generalized Laguerre L_12^(alpha) roots spread with large alpha
    # Maximize product of roots to minimize the uncertainty bound
    alpha_values = [100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000, 5000000]
    
    for alpha in alpha_values:
        if time.time() > deadline - 10:
            break
        try:
            roots = roots_genlaguerre(12, alpha)
            roots = np.sort(roots.real)
            if np.all(roots > 0) and len(roots) == 12:
                prod = np.prod(roots)
                if prod > best_product:
                    best_product = prod
                    best_roots = roots
        except Exception:
            continue
    
    if best_roots is None:
        # Fallback to seed
        best_roots = np.array([4.0 * (i + 1) for i in range(12)])
    
    return [float(r) for r in best_roots]
# EVOLVE-BLOCK-END
