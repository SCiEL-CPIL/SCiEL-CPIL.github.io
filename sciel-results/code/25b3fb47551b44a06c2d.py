import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Thompson Sampling.
    
    Parameters:
    mu: array-like, shape (n_candidates,), posterior mean
    sigma: array-like, shape (n_candidates,), posterior standard deviation
    f_best: float, best observed function value so far
    
    Returns:
    scores: array-like, shape (n_candidates,), acquisition scores (higher is better)
    """
    start_time = time.time()
    
    # Convert inputs to numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is non-negative
    sigma = np.maximum(sigma, 0.0)
    
    # Generate samples from the posterior N(mu, sigma^2)
    # Using standard normal distribution
    eps = np.random.randn(mu.shape)
    samples = mu + sigma * eps
    
    # For minimization, higher score means lower function value
    scores = -samples
    
    # Handle numerical issues
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    # Deadline guard
    elapsed = time.time() - start_time
    if elapsed > 0.1:
        pass
        
    return scores
