"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    """
    Modified Thompson Sampling acquisition function.
    
    Uses a discounted uncertainty term to balance exploration and exploitation.
    The discount factor (0.5) reduces the weight of the uncertainty, favoring
    exploitation of the mean prediction while still allowing some exploration.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        Mean of the GP posterior at each candidate point.
    sigma : array of shape (2000,)
        Standard deviation of the GP posterior at each candidate point.
    f_best : float
        Best (lowest) observed objective value so far.
        
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores; higher means more promising to evaluate next.
    """
    # Ensure sigma is positive to avoid numerical issues
    sigma = np.maximum(sigma, 1e-9)
    
    # Discount factor to control exploration level
    # A value less than 1.0 reduces exploration, favoring exploitation
    discount = 0.5
    
    # Draw one sample from each posterior Gaussian N(mu_i, (discount * sigma_i)^2)
    # This is equivalent to sampling from N(mu, sigma^2) and scaling the noise
    Z = np.random.standard_normal(len(mu))
    samples = mu + discount * sigma * Z
    
    # We want to minimize the objective, so higher acquisition score = lower sampled value
    # Return negative samples so that argmax(scores) corresponds to argmin(samples)
    return -samples
# EVOLVE-BLOCK-END
