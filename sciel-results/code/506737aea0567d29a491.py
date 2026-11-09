"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import roots_genlaguerre

def construct():
    # The task involves generalized Laguerre polynomials L_d^(alpha)(x) with alpha = -1/2.
    # We select the 12 double-root locations z_i as the roots of the Laguerre polynomial L_12^(-1/2)(x).
    # This choice is motivated by the fact that the basis functions are Laguerre polynomials,
    # and using the zeros of a related Laguerre polynomial often provides well-conditioned
    # and optimized node distributions for such interpolation/identity problems.
    
    # Compute the 12 roots of L_12^(-0.5)(x)
    # scipy.special.roots_genlaguerre(n, alpha) returns the roots and weights.
    # The roots are the first return value.
    roots, _ = roots_genlaguerre(12, -0.5)
    
    # Convert to list of floats
    z = [float(r) for r in roots]
    
    # The roots are positive, finite, and pairwise distinct.
    # They are returned in ascending order.
    
    return z
# EVOLVE-BLOCK-END
