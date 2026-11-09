import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Computes the Thompson Sampling acquisition function for minimization.
    
    Parameters:
    mu : array-like, shape (N,)
        Posterior mean at N candidate points.
    sigma : array-like, shape (N,)
        Posterior standard deviation at N candidate points.
    f_best : float
        Current best observed function value.
        
    Returns:
    scores : ndarray, shape (N,)
        Thompson Sampling scores for the candidate points.
    """
    # Ensure sigma is strictly positive to avoid issues
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Thompson Sampling: Draw a single scalar sample z from N(0,1)
    # and compute the posterior predictive sample at each candidate
    # sample = mu + sigma * z
    # We want to minimize, so we return negative of the sample
    # (higher score = better for selection)
    
    # Draw a single scalar z ~ N(0,1)
    z = np.random.standard_normal()
    
    # Compute posterior predictive samples
    samples = mu + sigma_safe * z
    
    # Return negative samples as scores (higher is better for selection)
    scores = -samples
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
