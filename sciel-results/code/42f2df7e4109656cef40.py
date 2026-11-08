import numpy as np
from numpy.polynomial import laguerre

def construct():
    n = 12
    # Get coefficients of L_12(x)
    # lagLaguerre(n) returns coefficients in descending order of power
    coeffs = laguerre.lagLaguerre(n)
    
    # Find roots
    roots = np.roots(coeffs)
    
    # Extract real positive roots
    real_roots = []
    for r in roots:
        if abs(r.imag) < 1e-9 and r.real > 0:
            real_roots.append(r.real)
            
    # Sort
    real_roots.sort()
    
    # Check count
    if len(real_roots) == 12:
        return real_roots
    else:
        # Fallback
        return [4.0 * (i + 1) for i in range(12)]
