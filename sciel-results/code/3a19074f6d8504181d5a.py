"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition function.
    
    Parameters:
    mu: array of shape (2000,) with posterior means.
    sigma: array of shape (2000,) with posterior standard deviations.
    f_best: float, the lowest observed value.
    
    Returns:
    array of shape (2000,) with scores. Higher score means higher priority.
    """
    # Ensure sigma is non-negative and finite to avoid numerical issues
    sigma = np.clip(sigma, 1e-9, None)
    
    # Generate standard normal samples
    z = np.random.randn(len(mu))
    
    # Sample from the posterior distribution N(mu, sigma)
    # y = mu + sigma * z
    samples = mu + sigma * z
    
    # We are minimizing the objective function.
    # Thompson Sampling selects the candidate with the lowest sampled value.
    # To fit the "higher score is better" interface, we return the negative of the sample.
    scores = -samples
    
    return scores
# EVOLVE-BLOCK-END
