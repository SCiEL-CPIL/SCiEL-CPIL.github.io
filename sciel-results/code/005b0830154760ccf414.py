import numpy as np
from numpy.polynomial import laguerre

def construct():
    """
    Returns the 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    Based on the Cohn-Goncalves construction, the optimal roots for the squared Laguerre polynomial 
    L_12(x)^2 are the roots of L_12(x).
    """
    # Compute the roots (nodes) of the Laguerre polynomial L_12(x)
    # laggauss(n) returns the nodes and weights for Gaussian quadrature,
    # where the nodes are the roots of L_n(x).
    roots, _ = laguerre.laggauss(12)
    
    # Ensure they are positive (they are by definition for Laguerre)
    # and return as a list of floats
    return [float(r) for r in roots]
