"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Lower Confidence Bound (LCB) acquisition function for minimization.
    
    The LCB criterion is: min (mu(x) - k * sigma(x))
    Since the evaluator selects the candidate with the HIGHEST score,
    we return the negative of the LCB value:
    score = -mu(x) + k * sigma(x)
    
    We use a fixed exploration constant k. A higher k encourages more exploration.
    """
    # Exploration constant. 
    # k=3.0 is suggested as "high" for aggressive exploration of multimodal landscapes.
    k = 3.0
    
    # Calculate LCB scores
    # mu and sigma are numpy arrays of shape (2000,)
    scores = -mu + k * sigma
    
    # Ensure finite values (sigma should be positive, mu finite)
    # If sigma is 0, score is -mu.
    # If there are NaNs or Infs, replace them.
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
