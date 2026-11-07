"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Ensure sigma is positive to avoid division by zero and log issues
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Calculate z-score: (f_best - mu) / sigma
    # High z means mu is significantly lower than f_best relative to uncertainty
    z = (f_best - mu) / sigma_safe
    
    # Probability of improvement
    p_improve = norm.cdf(z)
    
    # Uncertainty term: log(1 + sigma)
    # Using log scale to dampen the effect of very large sigma in unexplored regions
    # while still rewarding uncertainty
    unc_term = np.log1p(sigma_safe)
    
    # Hybrid Acquisition Function:
    # 1. p_improve * unc_term: Rewards regions likely to improve and have high uncertainty
    # 2. 0.5 * z: Rewards regions with high expected improvement (exploitation)
    # The coefficient 0.5 balances exploration and exploitation.
    
    score = p_improve * unc_term + 0.5 * z
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
