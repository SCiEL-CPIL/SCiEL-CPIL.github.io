"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def acquisition(mu, sigma, f_best):
    # Lower Confidence Bound (LCB) for minimization.
    # We want to maximize the score returned.
    # Minimization objective: minimize mu - beta * sigma.
    # Maximization score: beta * sigma - mu.
    
    beta = 1.5
    
    # Ensure sigma is non-negative and finite, though GP should provide this.
    # Clip sigma to avoid extreme values if necessary, but standard GP outputs are usually well-behaved.
    scores = beta * sigma - mu
    
    # Handle any potential numerical issues, though unlikely with standard GP.
    # If scores contain NaN or Inf, replace them with a very low value (unlikely to be selected).
    scores = np.nan_to_num(scores, nan=-1e10, posinf=1e10, neginf=-1e10)
    
    return scores
# EVOLVE-BLOCK-END
