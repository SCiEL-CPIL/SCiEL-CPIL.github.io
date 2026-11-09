"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    # Thompson Sampling approach.
    # We sample from the posterior distribution for each candidate.
    # The posterior for the function value at x is N(mu(x), sigma(x)^2).
    # We want to minimize f(x), so we look for the candidate with the lowest sampled value.
    # Since the evaluator picks the candidate with the HIGHEST score,
    # we return the negative of the sampled value.
    
    # Ensure sigma is non-negative and finite
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Draw independent standard normal samples for each candidate
    z = np.random.normal(0, 1, size=mu.shape)
    
    # Sampled function values
    sampled_values = mu + sigma_safe * z
    
    # Score is negative of sampled value (higher score = lower sampled value = better for minimization)
    scores = -sampled_values
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
