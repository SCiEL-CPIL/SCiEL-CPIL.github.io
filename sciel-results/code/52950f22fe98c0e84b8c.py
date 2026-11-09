"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np
import scipy.special

def construct():
    """
    Return 12 double-root locations z_1..z_12.
    Strategy: Use the zeros of the generalized Laguerre polynomial L_{12}^{(-1/2)}(x).
    These nodes are clustered near the origin, unlike the Bessel-zero-squared strategy,
    providing a structural change in the node distribution which may alter the
    quotient polynomial's largest root B.
    """
    # n = 12, alpha = -1/2
    n = 12
    alpha = -0.5
    
    # roots_genlaguerre returns the roots of L_n^(alpha)(x)
    # The roots are all real and positive for n >= 1 and alpha > -1
    roots = scipy.special.roots_genlaguerre(n, alpha)[0]
    
    # The roots are returned in ascending order, but we sort to be safe
    # and ensure they are floats for the interface
    z = [float(r) for r in sorted(roots)]
    
    # Verify validity: finite, > 0, distinct
    if len(z) != 12:
        return None
    for i in range(len(z)):
        if not math.isfinite(z[i]) or z[i] <= 0:
            return None
        for j in range(i+1, len(z)):
            if abs(z[i] - z[j]) < 1e-10:
                return None
                
    return z
# EVOLVE-BLOCK-END
