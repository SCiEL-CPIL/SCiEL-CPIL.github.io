import numpy as np
from scipy.special import roots_laguerre

def construct():
    """
    Returns the 12 positive double-root locations for the Laguerre combination
    (m=12, n=1) corresponding to the Cohn-Goncalves construction.
    These are the roots of the Laguerre polynomial L_12(x).
    """
    # m=12 is the degree of the Laguerre polynomial whose roots are the double roots
    roots, _ = roots_laguerre(12)
    # The roots are already positive and distinct. Sorting them for consistency.
    return sorted(roots.tolist())
