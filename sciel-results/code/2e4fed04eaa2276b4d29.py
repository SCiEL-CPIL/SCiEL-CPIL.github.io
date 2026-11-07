import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition function for minimization.
    
    Args:
        mu: array of shape (N,) posterior mean
        sigma: array of shape (N,) posterior standard deviation
        f_best: float, best observed value so far
        
    Returns:
        scores: array of shape (N,) higher is better
    """
    # Start time for deadline guard
    start_time = time.time()
    deadline = start_time + 325.0  # 5s margin before 330s limit

    # Draw standard normal samples
    # Using a fixed seed is not possible as we need fresh randomness each call
    # to avoid correlation between steps, but we don't have a seed parameter.
    # np.random is used.
    z = np.random.standard_normal(size=mu.shape)
    
    # Compute the posterior sample: y = mu + sigma * z
    # We want to minimize y, so higher score means lower y.
    # Score = -y
    y_sample = mu + sigma * z
    
    # Return negative sample as score (higher score -> lower y_sample -> better for minimization)
    scores = -y_sample
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    # Check deadline (unlikely to be hit, but good practice)
    if time.time() > deadline:
         # If we are running out of time, just return the current best effort
         # which is already computed. No further action needed as computation is done.
         pass
         
    return scores
