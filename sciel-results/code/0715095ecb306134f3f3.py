"""Cost-aware acquisition function for Bayesian optimization (styblinski tang 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware expected improvement with exploration bonus.
    
    Parameters:
    -----------
    mu : array_like, shape (2000,)
        GP mean predictions in objective units.
    sigma : array_like, shape (2000,)
        GP standard deviation predictions in objective units.
    f_best : float
        Best observed function value so far.
    cost : array_like, shape (2000,)
        Cost of evaluating each candidate.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure positive sigma and cost
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Standard Expected Improvement
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Exploration bonus: prefer high uncertainty regions
    # Normalize sigma to [0, 1] based on its range
    sigma_min = np.min(sigma)
    sigma_max = np.max(sigma)
    if sigma_max > sigma_min:
        sigma_norm = (sigma - sigma_min) / (sigma_max - sigma_min)
    else:
        sigma_norm = np.zeros_like(sigma)
    
    # Combine EI and exploration
    # Weight: 0.7 for EI, 0.3 for exploration
    weights_ei = 0.7
    weights_explore = 0.3
    
    # Normalize EI to [0, 1]
    ei_min = np.min(ei)
    ei_max = np.max(ei)
    if ei_max > ei_min:
        ei_norm = (ei - ei_min) / (ei_max - ei_min)
    else:
        ei_norm = np.zeros_like(ei)
    
    # Raw score: weighted combination
    raw_score = weights_ei * ei_norm + weights_explore * sigma_norm
    
    # Cost-aware: divide by cost to get score per unit cost
    scores = raw_score / cost
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
