"""Rewritten program for the suite task; structural change to critical points of L_24^(-1/2)."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np
import scipy.special

def construct():
    """
    Return 12 double-root locations z_1..z_12.
    
    Strategy: Use the stationary points (critical points) of the generalized Laguerre
    polynomial L_{24}^{(-1/2)}(x). By the identity L_n^{(alpha)'}(x) = -L_{n-1}^{(alpha+1)}(x),
    the critical points of L_{24}^{(-1/2)} are the zeros of L_{23}^{(1/2)}(x).
    This exploits the interlacing property between Laguerre zeros and their derivatives,
    providing a node distribution that is structurally different from simply using
    the zeros of L_{12}^{(-1/2)}.
    """
    # The critical points of L_{24}^{(-1/2)}(x) are the zeros of L_{23}^{(1/2)}(x)
    # since L_{24}^{(-1/2)'}(x) = -L_{23}^{(1/2)}(x)
    n_deriv = 23
    alpha_deriv = 0.5  # alpha + 1 = -1/2 + 1 = 1/2
    
    # roots_genlaguerre returns the roots of L_n^(alpha)(x)
    roots = scipy.special.roots_genlaguerre(n_deriv, alpha_deriv)[0]
    
    # The roots are returned in ascending order, but we sort to be safe
    # and ensure they are floats for the interface
    z = [float(r) for r in sorted(roots)]
    
    # We need exactly 12 roots. L_{23}^{(1/2)} has 23 roots, so we need to select 12.
    # We take the 12 smallest positive roots, as they are clustered near the origin
    # and should provide a good node distribution for the problem.
    if len(z) < 12:
        return None
    
    z = z[:12]
    
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
