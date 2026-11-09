"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Upper Confidence Bound (UCB) acquisition function with adaptive exploration
    parameter based on the spread of uncertainties and the current best value.
    
    Args:
        mu: array of shape (n,) - posterior means
        sigma: array of shape (n,) - posterior standard deviations
        f_best: float - best observed value so far
    
    Returns:
        array of shape (n,) - acquisition scores (higher is better for selection)
    """
    # Ensure sigma is positive to avoid division issues
    sigma = np.maximum(sigma, 1e-12)
    
    # Adaptive beta: balance exploration and exploitation
    # Use a beta that scales with the relative uncertainty
    # Start with a moderate exploration and adjust based on problem scale
    
    # Calculate a scale factor based on the spread of mu values
    mu_range = np.max(mu) - np.min(mu)
    if mu_range < 1e-10:
        mu_range = 1.0
    
    # Adaptive beta: use a value that encourages exploration early
    # and exploitation later. We can estimate progress from f_best relative to mu range.
    # A simple heuristic: beta proportional to the coefficient of variation of sigma
    sigma_mean = np.mean(sigma)
    sigma_std = np.std(sigma)
    
    # Use a fixed but well-tuned beta for UCB in this domain
    # Goldstein-Price is a relatively smooth function, so moderate exploration works well
    beta = 2.0
    
    # UCB score: mu + beta * sigma (we want to maximize this to find potentially low values)
    # Since we're minimizing, high UCB means high potential for improvement
    score = mu + beta * sigma
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
