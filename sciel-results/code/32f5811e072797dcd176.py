import numpy as np
import time

N = 400

def construct():
    """
    Returns a sequence of signed step heights.
    Strategy: Maximize the squared integral (sum of elements) while keeping 
    the max absolute autoconvolution as low as possible relative to that sum.
    
    The objective is max|C(k)| / (sum(a))^2.
    To minimize this, we want sum(a) to be as large as possible.
    The maximum possible sum for a sequence of +/-1 is N (all 1s).
    For all 1s, C(0)=N, max|C|=N, sum=N. Ratio = 1/N.
    This is the theoretical minimum for +/-1 sequences.
    
    We use a sequence of all 1s, with one -1 to ensure "signed" variation 
    if strictly required, though all 1s is the mathematical optimum for the 
    ratio if allowed.
    """
    # Start with all 1s
    a = np.ones(N, dtype=np.float64)
    
    # If "signed" implies both signs must be present, introduce one -1.
    # This reduces sum from N to N-2.
    # Sum^2 becomes (N-2)^2 ~ N^2.
    # Max autoconvolution remains ~ N.
    # Ratio remains ~ 1/N.
    a[0] = -1.0
    
    return a
