"""Rewritten program using UCB acquisition function."""
import numpy as np

# EVOLVE-BLOCK-START
def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Upper Confidence Bound (UCB) for minimization.
    
    Args:
        mu: Array of posterior means (shape 2000,).
        sigma: Array of posterior standard deviations (shape 2000,).
        f_best: The best (lowest) observed value so far.
        
    Returns:
        Array of scores (shape 2000,). Higher scores indicate more promising candidates.
    """
    # For minimization, we want to maximize the likelihood of finding a low value.
    # We use the lower bound of the confidence interval: mu - beta * sigma.
    # To make higher scores better, we negate this: -mu + beta * sigma.
    # Observation data suggests beta=1.0 performs well for this task.
    beta = 1.0
    
    scores = -mu + beta * sigma
    
    return scores
# EVOLVE-BLOCK-END
