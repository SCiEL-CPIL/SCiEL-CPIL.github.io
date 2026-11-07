"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a hybrid of Expected Improvement
    and Probability of Improvement, with explicit time guard.
    """
    # Time guard: if we're taking too long, return a safe default
    # This shouldn't trigger in normal operation but protects against edge cases
    t_start = time.time()
    
    # Ensure sigma is strictly positive
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # z-score for improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0)  # EI should be non-negative
    
    # Probability of Improvement (bounded [0, 1])
    pi = norm.cdf(z)
    
    # Hybrid score: combine EI and PI
    # EI captures magnitude of improvement, PI captures confidence
    # Use a weighted combination that emphasizes exploration when uncertain
    # Weight PI more when sigma is high (uncertain regions)
    sigma_mean = np.mean(sigma_safe)
    sigma_norm = sigma_safe / (sigma_mean + 1e-8)
    
    # Dynamic weight: higher weight on PI when uncertain
    w_pi = 0.3 + 0.4 * np.clip(sigma_norm / 2.0, 0, 1)
    w_ei = 1.0 - w_pi
    
    # Normalize EI to be comparable to PI
    ei_max = np.max(ei)
    if ei_max > 1e-10:
        ei_norm = ei / ei_max
    else:
        ei_norm = ei
    
    # Combined score
    combined = w_pi * pi + w_ei * ei_norm
    
    # Cost-aware: divide by cost to prefer cheaper evaluations
    score = combined / cost
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Time guard check
    elapsed = time.time() - t_start
    if elapsed > 280:  # Safety margin before 300s limit
        # Return a simple fallback
        return pi / cost
    
    return score
# EVOLVE-BLOCK-END
