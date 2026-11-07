import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition function for minimization.
    
    This function uses Thompson Sampling, which samples a value from the
    posterior predictive distribution for each candidate point and selects
    the point with the lowest sampled value. This is implemented by returning
    the negative of the sampled value as the score, so that the maximum
    score corresponds to the minimum sampled value.
    
    Args:
        mu: Array of posterior means (2000,)
        sigma: Array of posterior standard deviations (2000,)
        f_best: Current best observed value (scalar)
        
    Returns:
        Array of scores (2000,), higher is better.
    """
    # Ensure sigma is strictly positive to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Generate standard normal noise for each candidate
    epsilon = np.random.randn(mu.shape[0])
    
    # Sample from the posterior predictive distribution: y ~ N(mu, sigma^2)
    y_samples = mu + sigma_safe * epsilon
    
    # We want to minimize y, so we return -y_samples.
    # Higher score = Lower sampled value = Better candidate.
    scores = -y_samples
    
    # Ensure all scores are finite
    # If any score is non-finite (e.g., due to overflow), replace with a very low value
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
