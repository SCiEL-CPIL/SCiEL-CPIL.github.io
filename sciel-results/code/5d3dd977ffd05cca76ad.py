import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Probability of Improvement (PoI).
    
    Parameters:
    mu: (2000,) array of GP mean predictions.
    sigma: (2000,) array of GP standard deviation predictions.
    f_best: float, the best observed objective value so far.
    cost: (2000,) array of evaluation costs for each candidate.
    
    Returns:
    scores: (2000,) array of acquisition scores. Higher is better.
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    cost = np.asarray(cost)
    
    # Prevent division by zero
    eps = 1e-9
    sigma_safe = sigma + eps
    
    # Calculate z-score for Probability of Improvement
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement: P(f(x) < f_best)
    # This is the CDF of the standard normal at z
    poi = norm.cdf(z)
    
    # Cost-aware score: PoI / Cost
    # We want to maximize this score
    # If cost is very low, the score is boosted.
    scores = poi / (cost + eps)
    
    # Ensure finite outputs
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
