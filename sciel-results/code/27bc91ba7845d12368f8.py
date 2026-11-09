import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # UCB (Upper Confidence Bound) acquisition function
    # Score = -mu + beta * sigma
    # Higher score is better.
    # We minimize the function, so we want low mu.
    # -mu is high when mu is low.
    # sigma is high when uncertainty is high.
    # beta controls the exploration-exploitation trade-off.
    
    # Using a fixed beta. 1.5 is a common choice for balanced exploration.
    beta = 1.5
    
    scores = -mu + beta * sigma
    
    # Ensure scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
