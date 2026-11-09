import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a hybrid of Expected Improvement
    and Upper Confidence Bound (UCB) with adaptive parameters and an
    exponential cost penalty.
    
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
    
    # Adaptive exploration parameter for UCB
    # progress: 0 when f_best is large (early), 1 when f_best is close to 0 (late)
    # f_best is the minimum value, so it decreases.
    # For Rosenbrock, f_best >= 0.
    progress = 1.0 / (1.0 + f_best)
    
    # Beta for UCB: High exploration early, low exploration late
    beta = 1.0 * (1.0 - progress) + 0.2 * progress
    
    # Upper Confidence Bound
    ucb = f_best - mu + beta * sigma
    
    # Hybrid score: Weight EI more heavily, but keep UCB for exploration
    # This structure allows the algorithm to balance exploitation and exploration
    hybrid = 0.7 * ei + 0.3 * ucb
    
    # Adaptive cost penalty
    # Lambda increases as we make progress, making cost more important
    # This ensures that in the early stages, we explore broadly regardless of cost,
    # and in the late stages, we prefer cheaper evaluations.
    lambda_cost = 0.5 + 1.5 * progress
    
    # Exponential penalty based on cost relative to the minimum possible cost (1.0)
    # cost is in [1, 2], so (cost - 1) is in [0, 1]
    penalty = np.exp(-lambda_cost * (cost - 1.0))
    
    # Final score
    scores = hybrid * penalty
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
