"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Thompson Sampling acquisition function.
    
    Draws multiple Monte Carlo samples from the GP posterior at each candidate,
    computes the mean (expected value) of the samples, and normalizes by cost.
    Higher scores indicate candidates with lower expected cost-adjusted objective.
    """
    n_candidates = len(mu)
    n_samples = 50  # Number of Monte Carlo samples per candidate for stable estimates
    
    # Reshape for broadcasting: (n_samples, n_candidates)
    mu_b = mu.reshape(1, -1)
    sigma_b = sigma.reshape(1, -1)
    
    # Draw samples: (n_samples, n_candidates)
    rng = np.random.default_rng(42)
    samples = rng.normal(loc=mu_b, scale=sigma_b, size=(n_samples, n_candidates))
    
    # Compute expected value (mean of samples) for each candidate
    expected_values = np.mean(samples, axis=0)  # (n_candidates,)
    
    # Cost-aware score: lower expected value / lower cost is better
    # Score = -expected_value / cost (higher is better)
    scores = -expected_values / cost
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
