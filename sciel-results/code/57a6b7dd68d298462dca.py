import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition function for minimization.
    
    Args:
        mu: Mean of the GP posterior, shape (N,)
        sigma: Standard deviation of the GP posterior, shape (N,)
        f_best: Best observed value so far (unused in pure TS, but kept for interface)
        
    Returns:
        scores: Array of shape (N,) where higher scores indicate better candidates.
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Ensure sigma is non-negative and finite to avoid numerical issues
    sigma = np.maximum(sigma, 1e-9)
    
    # Draw a single standard normal random variable for each candidate
    # This simulates a single sample from the posterior predictive distribution
    # y_i ~ N(mu_i, sigma_i^2)
    # We use np.random.randn for efficiency
    z = np.random.randn(mu.shape[0])
    
    # Sampled values
    y_samples = mu + sigma * z
    
    # For minimization, we want to select the point with the lowest sampled value.
    # The acquisition score should be higher for better candidates.
    # So we negate the sampled value.
    scores = -y_samples
    
    # Ensure all scores are finite. If any input was NaN/Inf, handle it.
    # Though the problem states mu and sigma are in objective units and likely well-behaved,
    # we guard against overflow or invalid states.
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
