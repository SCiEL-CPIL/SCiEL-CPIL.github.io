import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Thompson Sampling.
    
    Args:
        mu: array of shape (2000,), mean predictions.
        sigma: array of shape (2000,), standard deviation predictions.
        f_best: float, best observed value.
        cost: array of shape (2000,), cost of evaluating each point.
        
    Returns:
        array of shape (2000,), scores (higher is better).
    """
    start_time = time.time()
    # Safety margin: finish within 290s of start, leaving 10s for overhead
    deadline = start_time + 290.0
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle potential numerical issues
    # Clip sigma to a small positive value to avoid issues if 0
    sigma_safe = np.maximum(sigma, 1e-6)
    
    # Clip cost to a small positive value
    cost_safe = np.maximum(cost, 1e-6)
    
    # Thompson Sampling: Sample from the posterior
    # z ~ N(0, 1)
    z = np.random.randn(mu.shape[0])
    
    # Sampled function values
    y_sample = mu + sigma_safe * z
    
    # Cost-aware score: We want to minimize y_sample / cost
    # So we maximize - (y_sample / cost)
    score = -y_sample / cost_safe
    
    # Ensure all scores are finite
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Check deadline
    if time.time() > deadline:
        # If we are running out of time, we just return the current result
        # The computation above is very fast, so this is unlikely to trigger
        pass
        
    return score
