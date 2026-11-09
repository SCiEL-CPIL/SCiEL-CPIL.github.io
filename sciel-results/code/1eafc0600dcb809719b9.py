"""Acquisition function for Bayesian optimization of Hartmann 3D."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Thompson Sampling.
    Samples from the posterior predictive distribution and selects
    the candidate with the lowest sampled value (highest score).
    """
    start_time = time.time()
    deadline = start_time + 2.5  # Safety margin
    
    # Guard against zero/negative sigma
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Thompson Sampling:
    # Draw a sample from the posterior for each candidate.
    # y ~ N(mu, sigma^2)
    # We want to minimize y, so we assign a higher score to lower y.
    # Score = -y
    
    # Generate random samples
    # Using a fixed seed is not possible as we don't have a state, 
    # but np.random is fine for stochastic acquisition.
    # To ensure reproducibility within a single call if needed, 
    # we could use a deterministic hash, but standard TS uses random.
    
    # For better performance and to avoid extreme outliers that might
    # cause numerical issues, we can clip the z-scores or use a 
    # deterministic approximation like LCB with a specific beta.
    # However, true Thompson Sampling is the structural change requested.
    
    # Let's use a standard normal sample
    z = np.random.normal(0, 1, size=mu.shape)
    
    # Sampled values
    y = mu + sigma_safe * z
    
    # Score: higher is better. We want the minimum y.
    # So score = -y
    scores = -y
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check deadline
    if time.time() > deadline:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
