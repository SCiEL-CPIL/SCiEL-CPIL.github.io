"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np
from scipy import special

def construct():
        """
        Return 12 double-root locations z_1..z_12.
        Strategy: Use the 12 positive zeros of the generalized Laguerre polynomial
        L_12^(-1/2)(x). These are the natural nodes for this problem structure
        involving alpha = -1/2, and may yield a smaller B value than Bessel-based
        approximations.
        """
        start_time = time.time()
        deadline = start_time + 5.0

        try:
            # Get the zeros of the generalized Laguerre polynomial L_12^(-1/2)(x)
            # scipy.special.roots_genlaguerre(n, alpha) returns the n zeros of L_n^(alpha)(x)
            n = 12
            alpha = -0.5
            z = special.roots_genlaguerre(n, alpha)
            
            # Convert to float list
            z = [float(zi) for zi in z]
            
            # Verify constraints: positive, finite, distinct
            if len(z) != 12:
                return [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
            
            # Check all are positive and finite
            for zi in z:
                if not (math.isfinite(zi) and zi > 0):
                    return [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
            
            # Check pairwise distinct
            z_sorted = sorted(z)
            for i in range(len(z_sorted) - 1):
                if abs(z_sorted[i+1] - z_sorted[i]) < 1e-15:
                    return [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
            
            return z
        except Exception:
            # Fallback: use evenly spaced positive values
            return [float(i) for i in range(1, 13)]
    # EVOLVE-BLOCK-END
