import numpy as np
from scipy.special import roots_laguerre

def construct():
    """
    Constructs the 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    Uses the zeros of the Laguerre polynomial L_12(x) as the optimal locations.
    """
    # Compute the roots (nodes) of the Laguerre polynomial L_12(x)
    # roots_laguerre(n) returns (nodes, weights)
    nodes, weights = roots_laguerre(12)
    
    # The nodes are already sorted in ascending order
    # Ensure they are positive (they are for Laguerre)
    roots = list(nodes)
    
    return roots
