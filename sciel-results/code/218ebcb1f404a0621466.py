import numpy as np
import time

# Track start time for deadline guard
_START_TIME = time.time()

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Thompson Sampling.
    
    Args:
        mu: (2000,) array of mean predictions.
        sigma: (2000,) array of standard deviation predictions.
        f_best: float, best observed value so far.
        cost: (2000,) array of evaluation costs.
        
    Returns:
        (2000,) array of scores. Higher scores are preferred.
    """
    # Deadline guard: If we are close to the 300s limit, return a fast, safe heuristic
    # to ensure we don't timeout. A greedy approach (-mu/cost) is fast and reasonable.
    elapsed = time.time() - _START_TIME
    if elapsed > 280.0:
        # Safety margin: return greedy scores
        # Handle division by zero or non-finite values
        cost_safe = np.where(cost > 1e-6, cost, 1.0)
        scores = -mu / cost_safe
        return np.where(np.isfinite(scores), scores, 0.0)

    # Thompson Sampling
    # Sample from the posterior predictive distribution N(mu, sigma)
    # We want to minimize the sampled value y.
    # To make it cost-aware, we minimize y / cost.
    # Since we return "higher is better", we return - (y / cost).
    
    # Ensure sigma is positive to avoid issues with sampling
    sigma_safe = np.maximum(sigma, 1e-6)
    
    # Generate random samples
    # np.random.randn generates standard normal variables
    samples = mu + sigma_safe * np.random.randn(len(mu))
    
    # Calculate cost-adjusted score
    # We want to pick the point with the lowest expected value per unit cost.
    # Score = - (Sampled Value / Cost)
    cost_safe = np.where(cost > 1e-6, cost, 1.0)
    scores = -samples / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
