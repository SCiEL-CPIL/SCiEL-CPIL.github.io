import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition function for Bayesian Optimization.
    
    Instead of using a fixed or heuristic UCB (Upper Confidence Bound), 
    this method samples a function value from the Gaussian posterior 
    N(mu, sigma^2) and selects the point with the minimum sampled value.
    
    This approach naturally balances exploration and exploitation:
    - High sigma (uncertainty) leads to larger variance in samples, 
      encouraging exploration in unexplored regions.
    - Low mu (good predicted value) increases the chance of being 
      the minimum, encouraging exploitation of promising regions.
    
    Parameters:
    -----------
    mu : numpy.ndarray
        Mean of the Gaussian process posterior, shape (n_candidates,)
    sigma : numpy.ndarray
        Standard deviation of the Gaussian process posterior, shape (n_candidates,)
    f_best : float
        Best observed function value (lowest)
        
    Returns:
    --------
    scores : numpy.ndarray
        Scores for each candidate, shape (n_candidates,). 
        Higher scores indicate better candidates to evaluate next.
        We return the negative of the sampled values, so higher is better.
    """
    # Ensure inputs are arrays
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    
    # Handle edge cases where sigma might be zero or very small
    # to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Sample from the posterior: f ~ N(mu, sigma^2)
    # We use a single random sample for each candidate
    samples = mu + sigma_safe * np.random.randn(len(mu))
    
    # For minimization, we want the minimum value.
    # The acquisition function should return higher scores for better candidates.
    # So we return the negative of the sampled values.
    scores = -samples
    
    return scores
