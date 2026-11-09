"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import roots_genlaguerre, eval_genlaguerre
import time

def construct():
    start_time = time.time()
    deadline = start_time + 600.0  # 600s wall-clock limit
    
    def time_check():
        return time.time() > deadline - 10.0
    
    # The task specifies m=12, n=1 for the Cohn-Goncalves Laguerre construction.
    # The "12 positive double roots" correspond to the roots of the generalized
    # Laguerre polynomial L_12^(1)(x), which are the extremal nodes for the
    # weight function x * e^(-x).
    #
    # The score is the "upper bound on C4", which in the context of Fourier
    # sign uncertainty and Laguerre constructions is typically related to
    # the last sign change or a norm of the constructed function.
    #
    # The generalized Laguerre polynomial L_n^(alpha)(x) has n positive roots.
    # For n=12, alpha=1, these roots are the optimal quadrature nodes for
    # the weight x * e^(-x), which is the natural weight for this problem.
    #
    # The "upper bound on C4" is likely the value of the constructed
    # polynomial or its norm at the last root, or simply a function of
    # the root locations. Given the problem statement says "computes the
    # last sign change", the score is likely the last root location or
    # a function thereof.
    #
    # Using the generalized Laguerre roots L_12^(1)(x) should give a
    # theoretically optimal or near-optimal configuration.
    
    try:
        # Get the 12 roots of the generalized Laguerre polynomial L_12^(1)(x)
        # alpha=1 corresponds to the weight x * e^(-x)
        roots, _ = roots_genlaguerre(12, 1)
        # Sort to ensure ascending order
        roots = np.sort(roots)
        # Ensure all roots are positive (they should be for Laguerre polynomials)
        if np.any(roots <= 0):
            # Fallback to standard Laguerre roots
            from scipy.special import roots_laguerre
            roots, _ = roots_laguerre(12)
            roots = np.sort(roots)
        
        return [float(r) for r in roots]
    
    except Exception:
        # Fallback: use standard Laguerre roots L_12(x)
        try:
            from scipy.special import roots_laguerre
            roots, _ = roots_laguerre(12)
            roots = np.sort(roots)
            return [float(r) for r in roots]
        except Exception:
            # Final fallback: arithmetic progression
            return [float(4.0 * (i + 1)) for i in range(12)]
# EVOLVE-BLOCK-END
