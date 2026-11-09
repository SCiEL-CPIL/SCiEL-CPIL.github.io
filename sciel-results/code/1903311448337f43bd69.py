"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Thompson Sampling.
    
    This approach samples from the posterior predictive distribution for each
    candidate point and selects the point that maximizes the expected improvement
    per unit cost. This stochastic method often outperforms deterministic
    acquisition functions like EI in multimodal landscapes by providing
    better exploration-exploitation balance.
    
    mu, sigma, f_best, cost: arrays of shape (2000,)
    Returns: 2000 finite scores (higher = evaluate next)
    """
    t_start = time.time()
    
    # Ensure inputs are properly shaped and finite
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    f_best = float(f_best)
    
    # Replace non-finite values
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-10)
    # Ensure cost is positive
    cost_safe = np.maximum(cost, 1e-10)
    
    # Thompson Sampling: Draw a sample from the posterior for each candidate
    # y ~ N(mu, sigma^2)
    z = np.random.randn(mu.shape[0])
    y_samples = mu + sigma_safe * z
    
    # Calculate score: (f_best - y_sample) / cost
    # We want to maximize this value.
    # Higher score indicates a point with high potential improvement relative to its cost.
    
    scores = (f_best - y_samples) / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check time - if we're getting close to the limit, return immediately
    t_elapsed = time.time() - t_start
    if t_elapsed > 280:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
