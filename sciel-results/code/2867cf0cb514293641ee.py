"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global variable to track the start time of the evaluation process
_start_time = None

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Thompson Sampling.
    
    Args:
        mu: Array of posterior means (shape 2000,).
        sigma: Array of posterior standard deviations (shape 2000,).
        f_best: The best observed value so far.
        
    Returns:
        Array of scores (shape 2000,). Higher scores indicate more promising candidates.
    """
    global _start_time
    
    # Initialize time on first call
    if _start_time is None:
        _start_time = time.time()
        
    elapsed = time.time() - _start_time
    
    # Deadline guard: If we are close to the 330s wall-clock limit,
    # switch to pure exploitation to ensure we utilize the remaining time
    # effectively to improve f_best, rather than exploring.
    # 290s provides a 40s safety margin.
    if elapsed > 290:
        # Return negative mean to prioritize candidates with lowest expected value
        return -mu
    
    # Thompson Sampling:
    # Draw a sample from the posterior distribution N(mu, sigma)
    # z ~ N(0, 1)
    z = np.random.normal(0, 1, size=mu.shape)
    
    # Sample from the posterior
    sample = mu + sigma * z
    
    # The evaluator selects the candidate with the highest score.
    # Since we are minimizing the objective, we return the negative of the sample
    # so that the minimum sampled value corresponds to the maximum score.
    return -sample
# EVOLVE-BLOCK-END
