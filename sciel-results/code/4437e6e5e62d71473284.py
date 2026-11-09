"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-Aware Expected Improvement with strict uncertainty penalty.
    
    In the sparse data regime, sigma is large everywhere, causing standard EI
    to be dominated by the uncertainty term. We apply a strict penalty that
    scales exponentially with sigma, forcing the acquisition to prioritize
    high-confidence improvements over risky exploration.
    
    The penalty is designed so that:
    1. When sigma is small (confident regions), the penalty is ~1 (minimal effect)
    2. When sigma is large (uncertain regions), the penalty approaches 0 (strong suppression)
    """
    # Small epsilon to avoid division by zero
    eps = 1e-10
    
    # Ensure sigma is positive and finite
    sigma_safe = np.maximum(sigma, eps)
    sigma_safe = np.where(np.isfinite(sigma_safe), sigma_safe, eps)
    
    # z-score for standard normal CDF/PDF
    # Clip z to avoid extreme values that cause numerical issues
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -10, 10)
    
    # Standard EI components
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei_mean_term = (f_best - mu) * cdf_z
    ei_uncertainty_term = sigma_safe * pdf_z
    
    # Strict uncertainty penalty: exp(-beta * sigma)
    # beta controls how aggressively we suppress high-sigma regions
    # With beta=2.0, sigma=0.5 gives penalty ~0.37, sigma=1.0 gives ~0.14
    beta = 2.0
    uncertainty_penalty = np.exp(-beta * sigma_safe)
    
    # Modified EI: mean improvement + strongly penalized uncertainty
    # The penalty ensures that only regions with low sigma contribute meaningfully
    # to the uncertainty term, while the mean term is always preserved
    modified_ei = ei_mean_term + uncertainty_penalty * ei_uncertainty_term
    
    # Ensure non-negative scores (EI should be non-negative)
    modified_ei = np.maximum(modified_ei, 0.0)
    
    # Cost-aware: divide by cost to get improvement per unit cost
    # cost = 1 + mean(x scaled to [0,1]), so higher cost means more expensive
    cost_safe = np.maximum(cost, eps)
    cost_safe = np.where(np.isfinite(cost_safe), cost_safe, eps)
    
    scores = modified_ei / cost_safe
    
    # Ensure all values are finite and non-negative
    scores = np.where(np.isfinite(scores), scores, 0.0)
    scores = np.maximum(scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
