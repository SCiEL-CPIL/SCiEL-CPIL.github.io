"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining UCB-style exploration with quantile-based
    diversification to avoid clustering of evaluations in high-uncertainty regions.
    
    The Beale function is multi-modal with a global minimum at (3, 0) with value 0.
    We need to balance:
    1. Exploitation: target regions with low predicted mean
    2. Exploration: visit regions with high uncertainty to discover new minima
    3. Diversification: spread evaluations across different quantiles of uncertainty
    """
    
    # Clip sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Compute standardized improvement (for EI)
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # UCB score: mu - beta * sigma (lower is better for minimization)
    # Use adaptive beta: larger when f_best is large (early), smaller when f_best is small (late)
    # f_best starts high (~15+) and decreases toward 0
    beta = 1.5 + 0.5 * np.exp(-f_best / 3.0)
    
    # UCB score (higher means more promising for exploration)
    ucb_score = -mu + beta * sigma_safe
    
    # Quantile-based diversification:
    # Partition candidates into bins based on their sigma (uncertainty)
    # Assign a bonus to each bin to encourage spreading across uncertainty levels
    n_candidates = len(sigma_safe)
    
    # Sort indices by sigma to create quantile bins
    sorted_indices = np.argsort(sigma_safe)
    n_bins = min(10, n_candidates)
    
    # Create bin assignments
    bin_sizes = np.full(n_bins, n_candidates // n_bins)
    remainder = n_candidates % n_bins
    for i in range(remainder):
        bin_sizes[i] += 1
    
    # Assign each candidate to a bin based on its rank in sigma
    bin_assignments = np.zeros(n_candidates, dtype=int)
    start = 0
    for i in range(n_bins):
        end = start + bin_sizes[i]
        bin_assignments[sorted_indices[start:end]] = i
        start = end
    
    # Compute the maximum sigma in each bin
    bin_max_sigma = np.zeros(n_bins)
    for i in range(n_bins):
        mask = bin_assignments == i
        if np.any(mask):
            bin_max_sigma[i] = np.max(sigma_safe[mask])
    
    # Normalize bin_max_sigma to [0, 1] for consistent scaling
    if np.max(bin_max_sigma) > 0:
        bin_max_sigma_norm = bin_max_sigma / np.max(bin_max_sigma)
    else:
        bin_max_sigma_norm = np.zeros(n_bins)
    
    # Diversification bonus: higher for candidates in bins with higher max uncertainty
    # This encourages visiting regions with high uncertainty, but spreads across bins
    div_bonus = 0.3 * bin_max_sigma_norm[bin_assignments]
    
    # Also add a small bonus for being in the top quantile of sigma (highest uncertainty)
    # to ensure we don't completely ignore the most uncertain regions
    sigma_rank = np.argsort(np.argsort(sigma_safe))  # rank from 0 to n-1
    top_uncertainty_bonus = 0.1 * (sigma_rank / n_candidates)
    
    # Combine scores:
    # Primary: UCB for exploration-exploitation balance
    # Secondary: EI for exploitation
    # Tertiary: Diversification to spread evaluations
    # 
    # Weight: when f_best is large, rely more on UCB/exploration
    # When f_best is small, rely more on EI/exploitation
    w_ucb = 0.6 + 0.4 * np.exp(-f_best / 5.0)
    w_ei = 1.0 - w_ucb
    
    # Normalize EI and UCB to comparable scales
    # EI can be large early, UCB is bounded by beta * max_sigma
    max_ei = np.max(ei) if np.max(ei) > 0 else 1.0
    max_ucb = np.max(ucb_score) if np.max(ucb_score) > 0 else 1.0
    
    ei_norm = ei / max_ei
    ucb_norm = ucb_score / max_ucb
    
    # Final score: higher means evaluate next
    scores = w_ucb * ucb_norm + w_ei * ei_norm + div_bonus + top_uncertainty_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Ensure non-negative
    scores = np.maximum(scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
