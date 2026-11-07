"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Adaptive acquisition function combining Expected Improvement (EI)
    and a UCB-like exploration term.

    mu: array of shape (N,) - GP mean at candidate points
    sigma: array of shape (N,) - GP standard deviation at candidate points
    f_best: float - best observed value so far (minimum)

    Returns: array of shape (N,) with scores (higher is better to explore)
    """
    # Ensure sigma is positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-9)

    # Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)

    # UCB term: balance exploitation and exploration
    # Use an adaptive beta that decreases as we get more confident
    # For simplicity, use a fixed beta that works well for this problem
    beta = 1.0
    ucb = mu + beta * sigma_safe  # Note: we want to minimize, so lower mu is better
    # Actually for minimization, UCB should be mu - beta*sigma (optimistic)
    # But since we're maximizing the acquisition function, let's think differently
    # We want to pick points where the function value is likely to be low
    # So a good acquisition would be: -mu + beta*sigma (exploit low mu, explore high sigma)

    # Let's use a combination:
    # Primary: EI (which already balances exploration/exploitation)
    # Secondary: Add a small exploration bonus for high uncertainty in regions
    # where the mean is not already very good

    # Scale EI to be comparable with other terms
    ei_max = np.max(ei)
    if ei_max > 0:
        ei_scaled = ei / ei_max
    else:
        ei_scaled = ei

    # Exploration term: favor points with high sigma but reasonable mu
    # Use a soft penalty for very high mu (poor exploitation)
    mu_range = np.max(mu) - np.min(mu)
    if mu_range > 0:
        mu_norm = (mu - np.min(mu)) / mu_range  # 0 = best, 1 = worst
    else:
        mu_norm = np.zeros_like(mu)

    # Exploration score: high sigma, low mu_norm
    sigma_max = np.max(sigma_safe)
    if sigma_max > 0:
        sigma_norm = sigma_safe / sigma_max
    else:
        sigma_norm = np.zeros_like(sigma_safe)

    # Combined score: primarily EI, with a small exploration boost
    # The exploration boost is strongest where sigma is high and mu is not terrible
    exploration_boost = 0.1 * sigma_norm * (1 - mu_norm * 0.5)

    # Final score
    scores = ei_scaled + exploration_boost

    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
# EVOLVE-BLOCK-END
