"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = None

def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition function.
    
    mu, sigma: (2000,) arrays, posterior mean and std in objective units.
    f_best: lowest observed value so far.
    Returns: (2000,) array of scores, higher = more promising to evaluate.
    """
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    # Safety margin: return a deterministic fallback if we're getting close to timeout
    elapsed = time.time() - _START_TIME
    if elapsed > 325.0:
        # Fallback: return -mu (pure exploitation) to be safe
        return -mu
    
    # Thompson Sampling: sample from the posterior for each candidate
    # For minimization, we want low values.
    # Sample: x ~ N(mu, sigma^2)
    # We want to pick the candidate with the lowest sampled value.
    # So score = -sample (higher score = lower sampled value = more promising)
    
    # Generate random samples
    # Use a fixed seed? No, Thompson Sampling requires randomness.
    # But we need to be careful: the evaluator calls this function multiple times.
    # Each call should use fresh randomness.
    
    # Handle edge cases where sigma might be 0 or very small
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Draw samples from N(mu, sigma_safe^2)
    samples = mu + sigma_safe * np.random.randn(len(mu))
    
    # For minimization, lower samples are better.
    # We return scores where higher is better, so we negate the samples.
    return -samples

# EVOLVE-BLOCK-END
