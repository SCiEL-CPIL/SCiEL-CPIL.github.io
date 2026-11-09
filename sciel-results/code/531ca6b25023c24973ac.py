"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses a hybrid of Expected Improvement (EI) and Upper Confidence Bound (UCB),
    both normalized by evaluation cost.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        Mean of the GP posterior
    sigma : array of shape (2000,)
        Standard deviation of the GP posterior
    f_best : float
        Best observed value so far
    cost : array of shape (2000,)
        Cost of evaluating each candidate
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores (higher = more promising)
    """
    # Ensure numerical stability
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)
    
    # 1. Calculate Expected Improvement (EI) for minimization
    # EI = E[max(f_best - f(x), 0)]
    z = (f_best - mu) / sigma_safe
    # Handle potential overflow/underflow in norm.cdf/pdf for extreme z
    # z > 10: cdf ~ 1, pdf ~ 0 -> EI ~ f_best - mu
    # z < -10: cdf ~ 0, pdf ~ 0 -> EI ~ 0
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # 2. Calculate UCB score for minimization
    # We want to minimize f(x). The UCB bound on the minimum is mu - beta * sigma.
    # To maximize the acquisition score (higher is better), we use:
    # Score_UCB = -(mu - beta * sigma) = beta * sigma - mu
    # Tuning beta: 2.0 is a standard choice for moderate exploration.
    beta = 2.0
    ucb_score = beta * sigma_safe - mu
    
    # 3. Normalize by cost
    # Cost is between 1.0 and 2.0.
    ei_score = ei / cost_safe
    ucb_score_norm = ucb_score / cost_safe
    
    # 4. Combine
    # Take the maximum to allow either exploitation (if EI is high) 
    # or exploration (if UCB is high) to drive the selection.
    scores = np.maximum(ei_score, ucb_score_norm)
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
