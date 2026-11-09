"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_START_TIME = time.time()
_TIME_LIMIT = 300.0  # 300 seconds limit, keep margin
_SAFETY_MARGIN = 5.0 # 5 seconds margin

def acquisition(mu, sigma, f_best):
    # Check time
    if time.time() - _START_TIME > _TIME_LIMIT - _SAFETY_MARGIN:
        # Return a fallback score, e.g., random or uniform
        # But we must return finite scores.
        # If we are out of time, we can't compute complex things.
        # Just return zeros or small randoms?
        # The prompt says "return the best result found". 
        # If we haven't computed anything, we return zeros.
        return np.zeros_like(mu)

    # UCB Acquisition Function
    # Minimize f(x). 
    # Predictive mean mu, std sigma.
    # We want high score for low mu and high sigma.
    # Score = -mu + kappa * sigma
    
    kappa = 1.5
    
    scores = -mu + kappa * sigma
    
    # Ensure finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
