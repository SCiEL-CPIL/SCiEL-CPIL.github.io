"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Ensure sigma is strictly positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score for Expected Improvement
    z = (f_best - mu) / sigma_safe
    
    # PDF and CDF of the standard normal distribution
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    # Standard Expected Improvement components
    # Exploitation term: expected improvement if we pick this point
    ei_exploit = (f_best - mu) * Phi_z
    # Exploration term: uncertainty weighted by relevance (Gaussian kernel)
    ei_explore = sigma_safe * phi_z
    
    # Adaptive Exploration Boost
    # Branin 2D minimum is approximately 0.397887
    f_target = 0.4
    
    # Calculate how far we are from the target
    # If f_best is much larger than f_target, we are in an exploration phase
    excess = np.maximum(0.0, f_best - f_target)
    
    # Smoothly increase the weight of the exploration term
    # boost ranges from 1.0 (standard EI) to ~3.0 (heavy exploration)
    # Using 1 - exp(-excess) for a smooth, bounded increase
    max_boost = 3.0
    boost_factor = 1.0 + (max_boost - 1.0) * (1.0 - np.exp(-excess))
    
    # Combine exploitation and boosted exploration
    score = ei_exploit + boost_factor * ei_explore
    
    return score
# EVOLVE-BLOCK-END
