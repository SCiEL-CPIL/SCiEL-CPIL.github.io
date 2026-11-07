"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Upper Confidence Bound (UCB) for minimization.
    
    Args:
        mu: Mean of the Gaussian Process posterior, shape (N,)
        sigma: Standard deviation of the Gaussian Process posterior, shape (N,)
        f_best: The best (lowest) observed function value so far.
        
    Returns:
        scores: Array of acquisition scores, shape (N,). Higher is better.
    """
    # UCB for minimization: we want to maximize (-mu + beta * sigma)
    # This encourages exploring regions with high uncertainty (sigma)
    # and exploiting regions with low predicted value (-mu).
    
    # Clip sigma to avoid numerical issues, though GP sigma should be positive
    sigma = np.maximum(sigma, 1e-8)
    
    # Beta controls the exploration-exploitation trade-off.
    # A larger beta encourages more exploration.
    # For multimodal functions like Michalewicz with a small budget,
    # a higher beta helps escape local minima.
    beta = 2.5
    
    scores = -mu + beta * sigma
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
