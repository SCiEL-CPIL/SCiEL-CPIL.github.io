"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Weierstrass 1D.
    
    Uses a dynamic UCB-style score that adapts the exploration-exploitation
    tradeoff based on the current state of optimization. For the Weierstrass
    function with only 30 total evaluations, we need aggressive exploration
    early to find the global minimum, then shift toward exploitation.
    
    The key insight: we don't know how many evaluations have been done, but
    we can infer progress from the relationship between mu, sigma, and f_best.
    When sigma is high relative to the spread of (f_best - mu), we're early;
    when sigma is low, we're later and should exploit more.
    
    Strategy:
    - Compute a base UCB score: -mu + kappa * sigma
    - Adapt kappa based on the typical uncertainty level
    - Add a small EI component for fine-tuning when exploitation is beneficial
    """
    # Ensure numerical stability
    sigma = np.maximum(sigma, 1e-10)
    
    # Compute expected improvement (for minimization)
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Compute a measure of overall uncertainty level
    # High median sigma suggests we're early in optimization
    med_sigma = np.median(sigma)
    med_diff = np.median(f_best - mu)
    
    # Normalize: if med_sigma is large relative to med_diff, we're uncertain/early
    # Use a smooth transition
    ratio = med_sigma / (np.abs(med_diff) + 1e-10)
    
    # Map ratio to a kappa value: early -> high kappa (explore), late -> lower kappa
    # When ratio is large (uncertain), use higher exploration
    # When ratio is small (confident), use lower exploration
    # Base kappa between 1.5 and 3.5
    kappa = 1.5 + 2.0 * np.exp(-0.5 * ratio)
    
    # UCB score for minimization
    ucb_score = -mu + kappa * sigma
    
    # EI score (already non-negative, higher is better)
    # Scale EI to be comparable to UCB
    ei_scale = np.abs(f_best - np.median(mu)) + 1e-10
    ei_score = ei / ei_scale
    
    # Combine: primarily UCB, with a small EI contribution
    # The EI helps fine-tune among points with similar UCB scores
    w_ei = 0.15
    score = (1.0 - w_ei) * ucb_score + w_ei * ei_score
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
