import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement with
    a progress-adaptive exponential cost penalty.
    """
    # Clip sigma to avoid division by zero or negative values
    sigma = np.maximum(sigma, 1e-12)
    cost = np.maximum(cost, 1e-12)
    
    # Standardized improvement
    z = (f_best - mu) / sigma
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Progress-based lambda: increases as f_best decreases
    # f_best is the minimum value, so it decreases toward 0.
    # lambda = 1 / (1 + f_best) gives higher penalty weight in late stages
    lambda_cost = 1.0 / (1.0 + f_best)
    
    # Exponential penalty based on cost relative to minimum possible cost (1.0)
    # cost is in [1, 2], so (cost - 1) is in [0, 1]
    penalty = np.exp(-lambda_cost * (cost - 1.0))
    
    # Final score: EI multiplied by cost penalty
    scores = ei * penalty
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
