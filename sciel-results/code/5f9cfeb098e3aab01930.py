import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    start_time = time.time()
    deadline = start_time + 290.0 # 40s safety margin from 330s limit
    
    # Upper Confidence Bound (UCB) for minimization
    # Score = mu - beta * sigma
    # We want to maximize this score.
    # This balances exploitation (low mu) and exploration (high sigma).
    beta = 2.0
    
    scores = mu - beta * sigma
    
    # Ensure all scores are finite
    # If any values are non-finite, replace with a very low score
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    # Check deadline
    if time.time() > deadline:
        return scores
        
    return scores
