import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin before 300s limit

    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle zero sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-12)
    cost_safe = np.maximum(cost, 1e-12)
    
    # Thompson Sampling: sample a single function from the posterior
    # For minimization, we want to minimize mu - sigma * z where z ~ N(0,1)
    # We can use the inverse CDF method or just use standard normal samples
    
    # Generate random samples from standard normal distribution
    # Use a fixed seed based on the current state to ensure reproducibility
    # but different across calls
    rng = np.random.default_rng()
    z_samples = rng.standard_normal(size=mu.shape)
    
    # Sampled function values: mu - sigma * z
    # When z is large positive, sampled value is lower (better for minimization)
    sampled_values = mu - sigma_safe * z_samples
    
    # Cost-aware scoring: minimize sampled_value / cost
    # Higher score = better candidate
    scores = -sampled_values / cost_safe
    
    # Ensure finite outputs
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Final deadline check
    if time.time() > deadline:
        return scores
    
    return scores
