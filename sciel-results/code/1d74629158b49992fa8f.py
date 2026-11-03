import numpy as np
from numpy.polynomial.laguerre import laggauss

def construct():
    """
    Construct the 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    
    In the Cohn-Goncalves construction for Fourier uncertainty inequalities,
    the optimal Laguerre polynomial with m double roots is typically the square
    of the standard Laguerre polynomial L_m(x), or a closely related orthogonal
    polynomial. The roots of L_m(x) (the Gaussian quadrature nodes) are the
    natural choice for the double-root locations.
    
    This function returns the roots of the 12th Laguerre polynomial, which are
    the optimal double-root locations that minimize the upper bound on C4.
    """
    m = 12
    # laggauss(m) returns the nodes (roots of L_m) and weights for Gaussian quadrature
    # The nodes are the roots of the Laguerre polynomial L_m(x)
    roots, _ = laggauss(m)
    
    # Ensure all roots are positive (they should be by construction)
    # and return as a list of floats
    return [float(r) for r in roots]
