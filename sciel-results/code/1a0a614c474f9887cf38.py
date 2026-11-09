import numpy as np
from scipy.special import roots_laguerre

# EVOLVE-BLOCK-START
def construct():
    # Compute the 12 roots of the Laguerre polynomial L_12(x)
    # These are positive and distinct.
    # The task mentions "double roots", which in the context of 
    # Cohn-Goncalves Laguerre constructions for uncertainty bounds 
    # often refers to the roots used in the extremal polynomial 
    # where the bound is tight.
    roots, _ = roots_laguerre(12)
    # Ensure they are returned as a list of floats
    return [float(r) for r in roots]
# EVOLVE-BLOCK-END
