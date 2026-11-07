import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Weierstrass 1D.
    Uses Upper Confidence Bound (UCB) for minimization.
    
    Parameters:
    mu (np.array): Posterior mean, shape (2000,)
    sigma (np.array): Posterior standard deviation, shape (2000,)
    f_best (float): Best observed function value so far.
    
    Returns:
    np.array: Scores for the candidates, shape (2000,). Higher is better.
    """
    start_time = time.time()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # UCB for minimization:
    # We want to minimize f(x).
    # The lower confidence bound is mu - beta * sigma.
    # We want to pick the point with the lowest LCB.
    # So the score (higher is better) is -LCB = -mu + beta * sigma.
    
    beta = 2.0
    
    scores = -mu + beta * sigma
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    # Time guard
    if time.time() - start_time > 290:
        return scores
        
    return scores
