import numpy as np
from scipy.special import roots_genlaguerre

def construct():
    """
    Return 12 double-root locations z_1..z_12.
    
    Strategy: Use the 12 positive zeros of the generalized Laguerre polynomial
    L_12^(-1/2)(x). These nodes are optimal for the weight x^(-1/2) exp(-x),
    which matches the orthogonality weight of the Laguerre basis used to construct g(x).
    This confines the interpolation to the main support of the weight,
    potentially reducing the largest root of the quotient q(x) compared to
    the asymptotic Bessel nodes which are scaled much larger.
    """
    # Get the roots of the generalized Laguerre polynomial L_12^(-1/2)(x)
    # n=12, alpha=-0.5
    nodes, _ = roots_genlaguerre(12, -0.5)
    
    # Ensure they are sorted (roots_genlaguerre returns them sorted)
    # They are all positive and distinct
    z_list = [float(z) for z in nodes]
    
    return z_list
