import numpy as np
import time

def acquisition(mu, sigma, f_best):
    start_time = time.time()
    deadline = start_time + 290.0 # 40s safety margin from 330s limit
    
    # Upper Confidence Bound (UCB) for minimization
    # score = mu - beta * sigma
    # Higher beta encourages more exploration.
    # Based on insights, beta=1.0 is a good balance for Ackley 2D.
    beta = 1.0
    
    scores = mu - beta * sigma
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check deadline
    if time.time() > deadline:
        return scores
        
    return scores
