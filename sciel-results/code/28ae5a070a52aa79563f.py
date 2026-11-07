import numpy as np
from scipy.stats import norm

# Global state to track evaluation count
_eval_count = 0

def acquisition(mu, sigma, f_best):
    global _eval_count
    _eval_count += 1
    
    # UCB for minimization: mu - beta * sigma
    # Higher score means more promising.
    # We want to minimize f, so we want low mu and high sigma (uncertainty).
    # Score = mu - beta * sigma.
    
    # Dynamic beta
    beta = np.sqrt(2 * np.log(_eval_count + 1))
    
    scores = mu - beta * sigma
    
    return scores
