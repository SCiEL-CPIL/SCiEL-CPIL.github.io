import numpy as np
from scipy.special import roots_laguerre

def construct():
    # The task asks for 12 positive double-root locations.
    # In the context of Laguerre polynomial constructions for uncertainty principles,
    # the relevant roots are often the zeros of the Laguerre polynomial L_m(x).
    # For m=12, we find the 12 roots of L_12(x).
    # These roots are positive and distinct.
    # The "double root" aspect might refer to the fact that in the squared polynomial 
    # or the specific construction, these locations are critical.
    # We return the roots of L_12(x).
    
    # scipy.special.roots_laguerre returns the roots of the Laguerre polynomial L_n(x)
    # which are the zeros of the polynomial.
    roots, _ = roots_laguerre(12)
    
    # Ensure they are positive (they are for Laguerre)
    # Sort them for consistency
    roots = np.sort(roots)
    
    return list(roots)
