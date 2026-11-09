import numpy as np
import numpy.polynomial.laguerre as npl
import time

def construct():
    start_time = time.time()
    deadline = 600.0
    
    # The roots of the Laguerre polynomial L_12(x)
    # laggauss(n) returns the nodes (roots) and weights for Gauss-Laguerre quadrature
    # The nodes are the roots of L_n(x)
    try:
        roots, weights = npl.laggauss(12)
        # Ensure they are positive and sorted
        roots = np.sort(roots).astype(float)
        return roots.tolist()
    except Exception:
        # Fallback to seed if something goes wrong
        return [4.0 * (i + 1) for i in range(12)]
