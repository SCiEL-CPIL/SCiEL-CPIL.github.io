import numpy as np
import scipy.special

def construct():
    """
    Construct 12 positive double-root locations for the Laguerre combination.
    Uses the roots of the Laguerre polynomial L_12(x) as the locations.
    """
    # roots_laguerre(n) returns the roots and weights of L_n(x)
    # We need 12 roots, so n=12
    roots, _ = scipy.special.roots_laguerre(12)
    
    # The roots are already positive and sorted in increasing order.
    # Convert to a list of floats for the interface.
    return [float(r) for r in roots]
