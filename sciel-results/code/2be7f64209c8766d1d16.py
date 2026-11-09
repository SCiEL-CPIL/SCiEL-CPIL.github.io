"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    """
    Return 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    Score is the upper bound on C4 (lower is better).
    """
    t0 = time.time()
    DEADLINE = 590  # safety margin before 600s kill
    
    m = 12
    n = 1
    
    # The task involves a refined Cohn-Goncalves Laguerre-polynomial construction.
    # The seed returns [4, 8, 12, ..., 48].
    # The score for the seed is 1.0.
    
    # Hypothesis: The score is related to the largest root r_m, normalized.
    # Or perhaps the score is directly a function of the roots that we need to minimize.
    
    # In uncertainty principles, bounds often depend on the maximum of a certain function.
    # For the Cohn-Goncalves construction, the bound is often related to the largest zero
    # of a specific orthogonal polynomial or a related extremal problem.
    
    # Let's consider the Laguerre polynomial L_m(x) or a related polynomial.
    # The zeros of L_12(x) are:
    # We can compute them using scipy or numpy.
    
    try:
        from scipy.special import roots_genlaguerre
        
        # The standard Laguerre roots (alpha=0) gave a score of 0.4869, which is worse than the seed (1.0).
        # This suggests that the optimal roots are not the standard Laguerre roots.
        # The seed roots [4, 8, ..., 48] are larger on average than the standard Laguerre roots.
        # Generalized Laguerre roots L_m^(alpha) for alpha > 0 are shifted to the right.
        # Let's try alpha = 1, which corresponds to the weight x * exp(-x).
        # This might better match the "n=1" dimension hint.
        
        x_gen, w_gen = roots_genlaguerre(m, alpha=1)
        
        # Verify positivity and order
        if not all(x > 0 for x in x_gen):
            raise ValueError("Non-positive roots")
            
        return [float(x) for x in x_gen]
        
    except Exception as e:
        # Fallback to seed if scipy fails or other issues
        return [4.0 * (i + 1) for i in range(12)]
# EVOLVE-BLOCK-END
