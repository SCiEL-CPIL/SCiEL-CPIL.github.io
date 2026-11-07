import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Thompson Sampling acquisition function.
    
    Parameters:
    mu: array-like, shape (2000,), mean of GP
    sigma: array-like, shape (2000,), std of GP
    f_best: float, best observed value
    cost: array-like, shape (2000,), cost of evaluating each point
    
    Returns:
    scores: array, shape (2000,), acquisition scores (higher is better)
    """
    start_time = time.time()
    deadline = 290.0  # Safety margin before 300s limit
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 1e-12), cost, 1.0)
    
    # Clip sigma to prevent numerical issues
    sigma = np.clip(sigma, 1e-8, 1e6)
    
    # Thompson Sampling: sample a single objective value from the Gaussian posterior
    # For each candidate, sample y ~ N(mu, sigma^2)
    # The sampled value represents what we'd observe if we evaluated at that point
    # We want to minimize, so lower sampled values are better
    # Score is based on how much better the sampled value could be vs f_best, adjusted for cost
    
    # Sample from N(mu, sigma)
    # Use a deterministic seed-like approach for reproducibility within a call
    # Actually, we just need one sample per candidate
    rng = np.random.default_rng()
    samples = mu + sigma * rng.standard_normal(len(mu))
    
    # The potential improvement is f_best - sample (positive if sample < f_best)
    # But we want to encourage exploration too, so we use the sample directly
    # Lower sample = better (more likely to find a good point)
    # Score = (f_best - sample) / cost, but this can be negative
    # Better: use the sample as a proxy, lower is better, so score = -sample / cost
    # Or: score = (f_best - sample) / cost to emphasize improvement
    
    # Thompson Sampling for minimization:
    # We sample y ~ N(mu, sigma^2) for each candidate
    # We pick the candidate with the lowest y (most likely to be good)
    # But we need to account for cost: prefer lower cost for same expected improvement
    
    # Score = -sampled_value / cost (higher is better)
    # This way, lower sampled values (better expected outcomes) get higher scores
    # And lower costs get higher scores
    
    scores = -samples / cost
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
