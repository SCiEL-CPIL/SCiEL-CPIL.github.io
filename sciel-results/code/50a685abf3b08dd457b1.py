"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Blends Expected Improvement (EI) with a PoI-like term and an uncertainty
    bonus to balance exploitation, exploration, and cost efficiency.
    """
    # Safety: ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standardized distance from best
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (EI)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Probability of Improvement (PoI)
    poi = norm.cdf(z)
    
    # Uncertainty bonus: encourages exploration of high-sigma regions
    # Scale by a factor relative to the current best uncertainty scale
    sigma_mean = np.mean(sigma_safe)
    uncertainty_bonus = sigma_safe / (sigma_mean + 1e-9)
    
    # Cost efficiency: prefer cheaper evaluations
    # Normalize cost to be in a reasonable range for weighting
    cost_safe = np.maximum(cost, 1e-9)
    
    # Blend EI and PoI for robustness
    # Use a weighted combination: 0.7 * EI + 0.3 * PoI * (f_best - min(mu))
    # The PoI term is scaled by the potential improvement range
    min_mu = np.min(mu)
    poi_scaled = poi * (f_best - min_mu + 1e-9)
    
    # Base acquisition: blend of EI and scaled PoI
    base = 0.7 * ei + 0.3 * poi_scaled
    
    # Add uncertainty bonus with a small weight
    score = base + 0.1 * uncertainty_bonus
    
    # Divide by cost for cost-awareness
    score = score / cost_safe
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
