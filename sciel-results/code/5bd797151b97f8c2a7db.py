"""Refined Cohn-Goncalves Laguerre construction for Fourier sign uncertainty."""
import numpy as np
from scipy.special import roots_laguerre

def construct():
    """
    Construct 12 positive double-root locations for the Laguerre combination
    (m=12, n=1) that minimize the C4 upper bound.
    
    The optimal double-root locations are the zeros of the 12th Laguerre polynomial,
    which are known to provide the optimal configuration for the Cohn-Goncalves
    construction in this setting.
    """
    # Compute zeros of the 12th Laguerre polynomial L_12(x)
    # These are the optimal double-root locations for m=12, n=1
    zeros, weights = roots_laguerre(12)
    
    # Return as a list of positive floats
    return [float(z) for z in zeros]
