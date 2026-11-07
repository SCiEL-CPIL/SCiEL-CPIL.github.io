"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_call_count = 0

def acquisition(mu, sigma, f_best):
    global _call_count
    _call_count += 1
    
    # Check time limit to ensure we return within constraints
    if time.time() - _start_time > 320:
        # Fallback: pure exploitation
        return -mu
    
    # Thompson Sampling: sample from the posterior and select the lowest
    # This inherently balances exploration and exploitation
    # Sample from N(mu, sigma^2) for each candidate
    # The candidate with the lowest sampled value gets the highest score
    
    # Ensure sigma is non-negative and finite
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Draw a random sample from the posterior for each candidate
    # Use a fixed seed based on call count for reproducibility within a run
    # but different across calls to maintain exploration
    rng = np.random.RandomState(_call_count * 1000 + 42)
    samples = mu + sigma_safe * rng.standard_normal(len(mu))
    
    # Score: higher is better, so we want the lowest sampled value
    # Negate the samples so that lower function values get higher scores
    scores = -samples
    
    # Ensure finite scores
    scores[~np.isfinite(scores)] = 0.0
    
    return scores
# EVOLVE-BLOCK-END
