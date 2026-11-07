"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = None

def _get_elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME


def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: if we're close to the 330s limit, return safe default
    elapsed = _get_elapsed()
    if elapsed > 320:
        return np.zeros(len(mu), dtype=np.float64)

    # Cost-aware Thompson Sampling-like acquisition
    # Sample from the posterior: y = mu + sigma * z where z ~ N(0,1)
    # Score = -y / cost (maximize expected improvement per unit cost)
    # This is a structural change from UCB to a sampling-based method
    
    # Clip sigma to avoid numerical issues
    sigma_clipped = np.maximum(sigma, 1e-12)
    
    # Sample a single draw from the GP posterior for each candidate
    # Use a fixed seed per call to make it deterministic but varied
    # Actually, we want randomness to explore, but we need reproducibility
    # Use a random draw each time
    z = np.random.randn(len(mu))
    y_sample = mu + sigma_clipped * z
    
    # Expected improvement: f_best - y_sample (positive if y_sample < f_best)
    improvement = f_best - y_sample
    
    # Cost-aware score: improvement divided by cost
    # This directly optimizes improvement per unit cost
    # Add a small epsilon to avoid division by zero
    scores = improvement / (cost + 1e-8)
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
