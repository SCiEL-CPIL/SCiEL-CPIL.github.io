import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of goldstein price 2d.
    Uses Upper Confidence Bound (UCB) adapted for minimization.
    Score = -mu + beta * sigma
    Higher score indicates a more promising candidate for evaluation.
    """
    start_time = time.time()
    deadline = 300.0
    safety_margin = 10.0
    
    # Exploration parameter beta
    # Based on knowledge that reducing beta improved performance, 
    # we use a conservative value.
    beta = 1.0
    
    # Compute UCB scores for minimization
    # We want to maximize the score, so we use -mu + beta*sigma
    # This favors low mean values and high uncertainty
    scores = -mu + beta * sigma
    
    # Ensure all scores are finite
    # If sigma is 0 or mu is inf, handle gracefully
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    # Check deadline (unlikely to be triggered for vectorized ops)
    if time.time() - start_time > deadline - safety_margin:
        # In a more complex implementation, we might return a partial result
        # or a default result. Here, the computation is already done.
        pass
        
    return scores
