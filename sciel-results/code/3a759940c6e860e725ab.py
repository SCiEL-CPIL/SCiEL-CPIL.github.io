import numpy as np
import time

def acquisition(mu, sigma, f_best):
    start_time = time.time()
    deadline = start_time + 290.0 # 40s safety margin from 330s limit
    
    # Use Lower Confidence Bound (LCB) for minimization
    # LCB = mu - beta * sigma
    # This encourages exploration of high uncertainty regions while exploiting low mu
    # A beta of 2.0 is a common choice for 2D problems with Ackley function
    
    beta = 2.0
    scores = mu - beta * sigma
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check deadline
    if time.time() > deadline:
        return scores
        
    return scores
