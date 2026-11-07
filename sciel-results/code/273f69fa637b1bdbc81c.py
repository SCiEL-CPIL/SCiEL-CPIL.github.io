"""Thompson Sampling acquisition for Bayesian optimization of Michalewicz 2D."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition function.
    
    Samples from the posterior N(mu, sigma^2) at each candidate point
    and returns the negative of the sample, so that the candidate with
    the lowest sampled function value gets the highest score.
    
    Parameters:
    -----------
    mu : np.ndarray, shape (2000,)
        Posterior mean at each candidate point.
    sigma : np.ndarray, shape (2000,)
        Posterior standard deviation at each candidate point.
    f_best : float
        Best (lowest) observed function value so far.
    
    Returns:
    --------
    scores : np.ndarray, shape (2000,)
        Finite scores; higher means evaluate next.
    """
    # Ensure sigma is non-negative and has a small floor to avoid numerical issues
    sigma = np.maximum(sigma, 1e-10)
    
    # Draw one sample from the posterior N(mu, sigma^2) for each candidate
    samples = np.random.normal(mu, sigma)
    
    # Return negative samples so that lower function values yield higher scores
    scores = -samples
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
