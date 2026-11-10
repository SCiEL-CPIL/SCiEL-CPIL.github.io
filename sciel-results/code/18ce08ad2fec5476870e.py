import time
import math
import numpy as np
from fractions import Fraction

def construct():
    """
    Returns 12 double-root locations z_1..z_12.
    Strategy: Use nodes from a weighted orthogonal polynomial (Laguerre-like)
    to minimize the spectral radius of the interpolation operator,
    which should reduce the largest real root r of q(x).
    """
    start_time = time.time()
    deadline = 600.0 - 30.0  # 30s safety margin

    # The task specifies alpha = -1/2 for the generalized Laguerre polynomial L_d^(alpha).
    # The weight function for these is w(x) = x^alpha * exp(-x) = x^(-0.5) * exp(-x).
    # We want to place roots z_i to minimize the "tail" of the rational function q(x).
    
    # A standard approach to minimize the max norm of the error or the growth of the
    # interpolation operator is to use the zeros of the orthogonal polynomial associated
    # with the weight function on the interval [0, infinity).
    
    # The roots of the generalized Laguerre polynomial L_n^(alpha)(x) are known to be
    # optimal in various minimax senses for interpolation on [0, infinity) with weight x^alpha e^-x.
    
    # However, we have 12 roots. Let's try using the roots of L_12^(-0.5)(x).
    # We can compute these using numpy's polynomial roots or scipy.special.roots_genlaguerre.
    
    try:
        from scipy.special import roots_genlaguerre
        # n=12, alpha=-0.5
        # Note: roots_genlaguerre returns roots of L_n^(alpha)(x)
        roots, weights = roots_genlaguerre(12, -0.5)
        # Ensure roots are positive and distinct
        z_list = [float(r) for r in roots]
        
        # Verify conditions
        if any(not math.isfinite(z) or z <= 0 for z in z_list):
            # Fallback to simple spacing if something went wrong
            z_list = [1.0 + i * 0.5 for i in range(12)]
            
        # Sort as required by the contract (though roots_genlaguerre usually returns sorted)
        z_list.sort()
        
        # Check pairwise distinctness
        for i in range(len(z_list) - 1):
            if z_list[i] == z_list[i+1]:
                # Perturb if necessary
                z_list[i+1] += 1e-10
                
        return z_list

    except Exception as e:
        # Fallback: Geometric progression
        # This is a reasonable heuristic for spreading points over [0, infinity)
        # to avoid clustering near 0 or infinity.
        z_list = [math.exp(0.5 * i) for i in range(12)]
        z_list.sort()
        return z_list

# EVOLVE-BLOCK-END
