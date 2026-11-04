"""Refined Cohn-Goncalves Laguerre construction for Fourier sign uncertainty."""
import numpy as np
from scipy.special import roots_genlaguerre
import time

def construct():
    """Return 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    
    The optimal roots for minimizing the C4 upper bound are the zeros of the
    generalized Laguerre polynomial L_m^{(2n+m-1)}(x) with m=12, n=1,
    i.e., L_{12}^{(13)}(x).
    """
    # Compute zeros of L_{12}^{(13)}(x)
    # roots_genlaguerre(n, alpha) returns zeros of L_n^{(alpha)}(x)
    zeros = roots_genlaguerre(12, 13)[0]
    
    # Sort the zeros to ensure consistent ordering
    zeros = np.sort(zeros)
    
    # Return as a list of floats
    return [float(z) for z in zeros]
