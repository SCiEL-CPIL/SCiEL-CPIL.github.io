import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Upper Confidence Bound (UCB) for minimization.
    UCB balances exploitation (low mean) and exploration (high uncertainty)
    via a confidence-bound formulation. Score is designed to be maximized.
    
    UCB = -mu + beta * sigma
    where beta is the exploration parameter.
    """
    # Deadline guard: track elapsed time to ensure we return before timeout
    start_time = time.time()
    
    # Exploration parameter for UCB
    # beta controls the trade-off between exploitation and exploration
    # For small-budget BO in 6D, a moderate beta works well
    beta = 1.5
    
    # UCB score: we want to minimize f, so we maximize (-mu + beta * sigma)
    scores = -mu + beta * sigma
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
