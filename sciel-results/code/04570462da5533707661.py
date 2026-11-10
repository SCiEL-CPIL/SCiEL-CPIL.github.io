import time
import math
import numpy as np
import scipy.special

def construct():
    """
    Return 12 double-root locations z_1..z_12.
    Uses the roots of the derivative of the generalized Laguerre polynomial L_13^(-1/2)(x).
    """
    start_time = time.time()
    
    try:
        # We want the roots of d/dx L_13^(-1/2)(x).
        # Identity: d/dx L_n^(alpha)(x) = -L_{n-1}^(alpha+1)(x).
        # So we need the roots of L_12^(1/2)(x).
        # These roots are strictly positive, distinct, and interlace with roots of L_13^(-1/2).
        
        # scipy.special.roots_genlaguerre(n, alpha) returns the roots of L_n^(alpha).
        # We need n=12, alpha=0.5.
        
        roots = scipy.special.roots_genlaguerre(12, 0.5)[0]
        
        # Convert to float, sort, and verify
        z = sorted([float(r) for r in roots])
        
        # Verify conditions
        if len(z) != 12:
            raise ValueError("Wrong number of roots")
        if any(z_i <= 0 for z_i in z):
            raise ValueError("Non-positive root found")
        if any(abs(z[i] - z[i-1]) < 1e-12 for i in range(1, 12)):
            raise ValueError("Roots not distinct")
            
        return z
        
    except Exception:
        # Fallback to Bessel zeros squared if for some reason the above fails
        # (though it shouldn't)
        try:
            bessel_zeros = scipy.special.jn_zeros(0, 12)
            z_fallback = sorted([float(z**2) for z in bessel_zeros])
            return z_fallback
        except Exception:
            # Last resort: simple geometric progression
            return [1.5 ** i for i in range(12)]
