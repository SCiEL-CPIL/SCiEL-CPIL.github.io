import numpy as np
import time

# Track the start time of the entire process to enforce the deadline
_START_TIME = time.time()

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Thompson Sampling.
    
    Args:
        mu: array of shape (2000,), mean of the GP posterior.
        sigma: array of shape (2000,), standard deviation of the GP posterior.
        f_best: float, the lowest observed value.
        cost: array of shape (2000,), cost of evaluating each candidate.
        
    Returns:
        array of shape (2000,), scores where higher is better.
    """
    # Deadline guard: If we are close to the 330s limit, use a cheap, deterministic fallback
    # to ensure we can finish the current iteration and return a valid result.
    elapsed = time.time() - _START_TIME
    if elapsed > 300:
        # Fallback: Greedy cost-aware exploitation
        # We want to minimize mu / cost.
        # Score = - (mu / cost)
        safe_cost = np.maximum(cost, 1e-9)
        scores = - (mu / safe_cost)
        return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

    # Thompson Sampling
    # 1. Draw samples from the posterior N(mu, sigma)
    # Ensure sigma is positive to avoid division issues or zero variance
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Generate standard normal samples
    z = np.random.standard_normal(mu.shape)
    
    # Compute sampled objective values
    # v is the predicted objective value for each candidate under this sample
    v = mu + sigma_safe * z
    
    # 2. Calculate cost-aware score
    # We want to select the candidate with the lowest sampled value per unit cost.
    # Metric to minimize: v / cost
    # Score to maximize: - (v / cost)
    
    safe_cost = np.maximum(cost, 1e-9)
    scores = - (v / safe_cost)
    
    # 3. Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
