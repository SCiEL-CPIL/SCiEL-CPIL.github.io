import numpy as np
from numpy.polynomial.laguerre import laggauss, lagval

def construct():
    # Approach: Directly compute the roots of the Laguerre polynomial L_12(x).
    # The Cohn-Goncalves construction often relies on the orthogonality properties
    # of Laguerre polynomials. The "12 double roots" likely correspond to the
    # roots of L_12(x), which are the nodes for Laguerre quadrature.
    
    # laggauss(n) returns the nodes (roots) and weights for the n-point Laguerre quadrature.
    # The nodes are the roots of the Laguerre polynomial L_n(x).
    n = 12
    roots, weights = laggauss(n)
    
    # The roots are already positive and sorted.
    # Return as a list of floats.
    return [float(r) for r in roots]
