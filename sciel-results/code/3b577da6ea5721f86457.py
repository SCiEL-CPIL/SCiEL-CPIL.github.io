import numpy as np
from scipy.special import roots_laguerre

def construct():
    # Compute the 12 roots of the Laguerre polynomial L_12(x)
    # These are positive and distinct.
    # The task asks for "12 positive double-root locations".
    # This implies we return the locations r_i.
    roots, weights = roots_laguerre(12)
    # roots are already positive and sorted
    return list(roots)
