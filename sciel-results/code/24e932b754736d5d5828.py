"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def acquisition(mu, sigma, f_best, cost):
    # Ensure inputs are 1-D arrays of length 2000
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: replace non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 1e-12), cost, 1.0)
    
    # Clip sigma to avoid division by zero and extreme values
    sigma = np.clip(sigma, 1e-8, 1e6)
    
    # Thompson Sampling: sample from the Gaussian posterior for each candidate
    # Use a fixed seed based on the hash of inputs to ensure reproducibility
    # within a single call, but different across calls if needed
    # Actually, for BO, we want randomness to encourage exploration.
    # We'll use np.random default generator.
    samples = np.random.normal(mu, sigma)
    
    # We want to minimize f, so lower sample is better.
    # Score should be higher for better (lower) samples.
    # So we use negative of the sample.
    ts_score = -samples
    
    # Cost-aware: divide by cost
    # Higher cost should reduce the score.
    cost_aware_score = ts_score / cost
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(cost_aware_score), cost_aware_score, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
