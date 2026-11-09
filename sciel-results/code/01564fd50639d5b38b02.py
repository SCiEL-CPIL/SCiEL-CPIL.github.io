import numpy as np
from scipy.special import roots_genlaguerre

def construct():
    # The task specifies m=12 and n=1, corresponding to the generalized Laguerre polynomial L_12^(1)(x).
    # We use scipy.special.roots_genlaguerre to obtain the exact positive roots.
    # n=12 is the degree, alpha=1 is the parameter.
    roots, weights = roots_genlaguerre(12, 1)
    
    # Ensure the roots are returned as a list of floats
    return [float(r) for r in roots]
