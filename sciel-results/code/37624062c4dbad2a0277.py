"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Track start time for potential timeout guards if needed, 
# though simple vectorized ops are very fast.
_start_time = time.time()

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Thompson Sampling.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean of the GP posterior at candidate points.
    sigma : ndarray, shape (2000,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best (lowest) observed objective value so far.
        
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher values indicate more promising candidates.
        We return -sampled_values so that the point with the lowest sampled value 
        has the highest score.
    """
    # Generate a single random sample from the standard normal distribution
    # This sample is used to draw from the posterior predictive distribution 
    # N(mu, sigma^2) for all candidate points simultaneously.
    z = np.random.normal(0, 1)
    
    # The sampled value for each point is mu + sigma * z
    # Since we are minimizing, we want to select the point with the lowest sampled value.
    # The acquisition function returns scores where higher is better.
    # Therefore, we return the negative of the sampled values.
    sampled_values = mu + sigma * z
    return -sampled_values
# EVOLVE-BLOCK-END
