import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a hybrid of Expected Improvement
    and Upper Confidence Bound (UCB), normalized by evaluation cost.
    
    Parameters:
    -----------
    mu : ndarray, shape (n,)
        GP mean predictions.
    sigma : ndarray, shape (n,)
        GP standard deviation predictions.
    f_best : float
        Best observed objective value so far.
    cost : ndarray, shape (n,)
        Cost of evaluating each candidate.
    
    Returns:
    --------
    scores : ndarray, shape (n,)
        Acquisition scores (higher = more desirable).
    """
    # Clip sigma to avoid division by zero or negative values
    sigma = np.maximum(sigma, 1e-12)
    cost = np.maximum(cost, 1e-12)
    
    # Standardized improvement
    z = (f_best - mu) / sigma
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Upper Confidence Bound: balances exploitation and exploration
    # beta controls the exploration-exploitation tradeoff
    beta = 0.5
    ucb = f_best - mu + beta * sigma
    
    # Combine EI and UCB with equal weight
    # EI captures expected improvement, UCB captures uncertainty
    combined = 0.5 * ei + 0.5 * ucb
    
    # Cost-aware score: normalize by evaluation cost
    scores = combined / cost
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
