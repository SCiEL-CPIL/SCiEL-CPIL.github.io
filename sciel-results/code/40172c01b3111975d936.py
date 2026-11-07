import numpy as np
import time

# Global timer for deadline guard
_start_time = time.time()
_TIME_LIMIT = 300.0  # Seconds

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses Cost-Weighted Upper Confidence Bound (UCB) for minimization.
    """
    # Deadline guard: if we are close to the time limit, fall back to exploitation
    elapsed = time.time() - _start_time
    if elapsed > _TIME_LIMIT - 5.0:
        # Return scores based on predicted mean (exploitation) to ensure valid finite output
        # Higher score = lower mu (since we minimize)
        return -mu

    # Ensure sigma is positive to avoid division by zero or negative variance issues
    sigma = np.maximum(sigma, 1e-9)
    
    # Ensure cost is positive
    cost = np.maximum(cost, 1e-9)
    
    # UCB for minimization:
    # We want to maximize the acquisition score.
    # A good candidate has low predicted mean (mu) and high uncertainty (sigma).
    # Score = -mu + beta * sigma
    # We divide by cost to account for the expense of evaluation.
    
    # Beta controls the exploration/exploitation trade-off.
    # A value of 1.5 is a reasonable default for balancing both.
    beta = 1.5
    
    ucb_scores = -mu + beta * sigma
    
    # Cost-aware scores
    scores = ucb_scores / cost
    
    # Handle numerical instabilities
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
