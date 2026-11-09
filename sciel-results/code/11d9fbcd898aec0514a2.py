"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best):
    # Thompson Sampling: sample from posterior for each candidate
    # This naturally balances exploration and exploitation
    # Use a fixed random seed for reproducibility within a single call
    # but different seeds across calls to get different samples
    
    # Use the current time as a seed to get different samples each call
    # but ensure it's deterministic enough for the evaluation
    # Actually, we need to be careful: the evaluator calls this multiple times
    # We want different samples each time to explore
    
    # Simple approach: use a random seed based on time or just use np.random
    # Since we need 2000 samples, let's use np.random.normal
    # To ensure we get different samples each call, we can use a global state
    # or just rely on np.random's internal state
    
    # Generate random normal samples
    samples = np.random.normal(0, 1, size=(2000,))
    
    # Thompson sampling score: sample from posterior
    # Higher score means more promising (lower function value expected)
    # So we want to minimize the sampled value
    # score = -(mu + sigma * samples) to make higher score = better
    
    # But wait: the task says "higher meaning evaluate next"
    # And we're minimizing, so we want candidates with low expected values
    # Thompson sampling: sample f(x) ~ N(mu, sigma^2)
    # We want to select x that minimizes the sampled value
    # So score should be such that higher score = lower sampled value
    # score = -sampled_value = -(mu + sigma * samples)
    
    score = -(mu + sigma * samples)
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
# EVOLVE-BLOCK-END
