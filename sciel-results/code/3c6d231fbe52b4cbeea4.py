"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import roots_genlaguerre

def construct():
    # The task specifies m=12 and n=1.
    # In the context of Laguerre polynomials L_m^(alpha), 
    # and given "dimension n=1", it is highly probable that alpha=1.
    # The "double root" construction in Cohn-Goncalves often aligns with 
    # the roots of the generalized Laguerre polynomial with alpha=1.
    
    m = 12
    alpha = 1
    
    # Compute the roots of L_12^(1)(x)
    roots, _ = roots_genlaguerre(m, alpha)
    
    # Ensure they are positive (they should be for alpha > -1)
    # and return as a list of floats
    return [float(r) for r in roots]
# EVOLVE-BLOCK-END
