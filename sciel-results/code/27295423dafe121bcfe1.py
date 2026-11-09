import numpy as np
from scipy.stats import truncnorm

def acquisition(mu, sigma, f_best):
    """
    Truncated Thompson Sampling acquisition function.
    
    Samples from the GP posterior y ~ N(mu, sigma^2) truncated to [0, inf),
    respecting the known constraint that the Beale function is non-negative.
    This avoids the discrete spike at 0 created by hard clipping and provides
    a smooth ranking of candidates based on their likely minimum values.
    """
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Calculate truncation points in standard normal space
    # y = mu + sigma * z
    # We want y >= 0 => z >= -mu / sigma
    a = -mu / sigma_safe
    b = np.inf
    
    # Sample from the truncated normal distribution
    # size=mu.shape ensures we get an array of the same shape
    samples = truncnorm.rvs(a, b, loc=mu, scale=sigma_safe, size=mu.shape)
    
    # Numerical safety: ensure samples are non-negative
    samples = np.maximum(samples, 0.0)
    
    # Higher score is better. We want to minimize objective, so score = -sample
    return -samples
