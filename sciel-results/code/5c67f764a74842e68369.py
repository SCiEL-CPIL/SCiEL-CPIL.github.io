import numpy as np
from scipy.special import roots_laguerre

def construct():
    """
    Returns the 12 roots of the Laguerre polynomial L_12(x).
    These roots are the optimal locations for the double roots in the
    Cohn-Goncalves construction, minimizing the associated uncertainty bound.
    """
    # Get the roots of the Laguerre polynomial of degree 12
    # roots_laguerre returns (roots, weights)
    roots, _ = roots_laguerre(12)
    
    # Ensure they are returned as a list of floats
    # The roots are already positive and distinct
    return [float(r) for r in roots]
