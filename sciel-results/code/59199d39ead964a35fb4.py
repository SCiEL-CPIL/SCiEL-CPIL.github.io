import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Thompson Sampling for Bayesian optimization.
    
    Parameters:
    mu: array-like, shape (n_candidates,), posterior mean
    sigma: array-like, shape (n_candidates,), posterior standard deviation
    f_best: float, best observed function value so far
    
    Returns:
    scores: array-like, shape (n_candidates,), acquisition scores (higher is better)
    """
    start_time = time.time()
    
    # Thompson Sampling: sample from the posterior
    # For minimization, we want to select points with low predicted values
    # Sample f ~ N(mu, sigma^2)
    # Return samples as scores (higher means we prefer to evaluate there)
    # Since we're minimizing, we want low samples, so we negate
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is non-negative
    sigma = np.maximum(sigma, 1e-10)
    
    # Generate random normal samples
    n = len(mu)
    z = np.random.standard_normal(n)
    
    # Sample from the posterior
    samples = mu + sigma * z
    
    # For minimization, we want to select points with the LOWEST samples
    # So we negate to make higher scores better
    scores = -samples
    
    # Handle numerical issues
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    # Deadline guard
    elapsed = time.time() - start_time
    if elapsed > 0.5:
        # If somehow this is slow, we still return the computed scores
        pass
    
    return scores
