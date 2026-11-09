# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    """
    Pure Thompson Sampling acquisition function.
    Samples from the GP posterior and selects candidates based on
    the sampled value divided by cost (negated for minimization).
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive to avoid issues
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Draw standard normal random variables for Thompson Sampling
    z = np.random.standard_normal(mu.shape)
    
    # Sample from the posterior distribution N(mu, sigma^2)
    sampled_values = mu + sigma_safe * z
    
    # For minimization, we prefer low sampled values.
    # Score is higher for better candidates, so we negate the sampled value.
    # We divide by cost to penalize expensive evaluations.
    scores = -sampled_values / cost_safe
    
    # Handle any non-finite values (e.g., from overflow or invalid inputs)
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
