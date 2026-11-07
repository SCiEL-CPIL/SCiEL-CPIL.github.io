"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition function.
    
    Args:
        mu: array of shape (2000,) with posterior mean.
        sigma: array of shape (2000,) with posterior standard deviation.
        f_best: float, the best objective value found so far.
        
    Returns:
        array of shape (2000,) with scores. Higher scores are preferred.
    """
    # Clip sigma to avoid negative or zero variance issues, though sigma should be >= 0
    # Using a small epsilon to prevent division by zero if used elsewhere, 
    # but here we just multiply, so 0 is fine. 
    # However, if sigma is 0, the sample is just mu.
    
    # Generate standard normal random variables
    z = np.random.randn(mu.shape[0])
    
    # Sample from the posterior distribution: y ~ N(mu, sigma^2)
    y_sample = mu + sigma * z
    
    # We want to minimize y, so we maximize -y
    return -y_sample
# EVOLVE-BLOCK-END
