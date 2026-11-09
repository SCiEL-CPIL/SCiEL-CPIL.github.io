"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Hybrid acquisition function for Bayesian optimization of Michalewicz 2D.
    Combines Expected Improvement (EI) with a UCB component, using an adaptive
    beta that balances exploration and exploitation based on the current
    posterior statistics and the observed best value.
    """
    # Clip sigma to avoid numerical issues
    sigma = np.maximum(sigma, 1e-10)
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z) where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    # Clip z to avoid extreme values
    z = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)  # EI is non-negative
    
    # UCB score for minimization: -mu + beta * sigma
    # We need to choose an adaptive beta
    
    # Compute a normalized exploration bonus
    # Use the ratio of sigma to the range of mu as a guide
    mu_range = np.max(mu) - np.min(mu)
    if mu_range < 1e-10:
        mu_range = 1.0
    
    # Base beta based on the current improvement potential
    # When f_best is much better than the mean, we should exploit more
    # When f_best is close to the mean, we should explore more
    improvement_ratio = np.maximum(f_best - np.min(mu), 0.0) / (mu_range + 1e-10)
    
    # Beta decreases as improvement ratio increases (exploit when close to optimum)
    # Beta increases when improvement ratio is low (explore when far from optimum)
    beta = 0.5 + 1.5 * (1.0 - np.clip(improvement_ratio, 0.0, 1.0))
    
    # Also consider the uncertainty level
    # High uncertainty regions get more exploration
    sigma_mean = np.mean(sigma)
    sigma_max = np.max(sigma)
    if sigma_max > 1e-10:
        uncertainty_ratio = sigma_mean / sigma_max
    else:
        uncertainty_ratio = 0.0
    
    # Adjust beta slightly based on uncertainty distribution
    beta = beta * (0.8 + 0.4 * (1.0 - uncertainty_ratio))
    
    # Compute UCB scores
    ucb_scores = -mu + beta * sigma
    
    # Combine EI and UCB
    # Normalize both to have similar scales
    ei_max = np.max(ei)
    ucb_max = np.max(np.abs(ucb_scores))
    
    if ei_max > 1e-10:
        ei_norm = ei / ei_max
    else:
        ei_norm = np.zeros_like(ei)
    
    if ucb_max > 1e-10:
        ucb_norm = ucb_scores / ucb_max
    else:
        ucb_norm = np.zeros_like(ucb_scores)
    
    # Weighted combination: 70% EI, 30% UCB
    scores = 0.7 * ei_norm + 0.3 * ucb_norm
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
