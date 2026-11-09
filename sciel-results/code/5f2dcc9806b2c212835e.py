import numpy as np
from scipy.special import roots_genlaguerre

def construct():
    # We need 12 double roots.
    # The basis is L_0, L_2, ..., L_50.
    # A good guess for the roots z_i is the roots of the Laguerre polynomial
    # of degree 24 with alpha = -1/2.
    # We take the 12 smallest positive roots.
    
    # Calculate roots of L_24^(-1/2)(x)
    # n=24, alpha=-0.5
    x, w = roots_genlaguerre(24, -0.5)
    
    # The roots are already sorted in ascending order by scipy
    # We take the first 12
    z = x[:12]
    
    # Ensure they are Python floats
    return [float(z_i) for z_i in z]
