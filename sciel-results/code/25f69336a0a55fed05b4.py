import numpy as np
import scipy.special as sp

def construct():
    """
    Returns the 12 smallest positive zeros of the generalized Laguerre polynomial
    L_12^(-1/2)(x). These are the Gauss-Laguerre quadrature nodes for alpha=-1/2.
    This choice is theoretically optimal for minimizing the norm of the interpolation
    error and is expected to reduce the magnitude of the remaining roots in q(x)
    compared to the uniform grid used in the seed.
    """
    n = 12
    alpha = -0.5
    
    # Get the coefficients of the generalized Laguerre polynomial L_n^(alpha)(x)
    # scipy.special.genlaguerre returns coefficients in descending order of powers
    coeffs = sp.genlaguerre(n, alpha)
    
    # Find the roots
    roots = np.roots(coeffs)
    
    # Filter for real, positive roots
    # Due to numerical precision, imaginary parts should be very small
    real_roots = []
    for r in roots:
        if abs(np.imag(r)) < 1e-10:
            real_part = np.real(r)
            if real_part > 1e-10:
                real_roots.append(real_part)
    
    # We expect exactly 12 positive real roots
    if len(real_roots) != 12:
        # Fallback to seed if something goes wrong (should not happen)
        return [4.0 * (i + 1) for i in range(12)]
        
    # Sort the roots
    real_roots.sort()
    
    return real_roots
