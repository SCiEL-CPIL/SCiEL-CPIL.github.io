"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Thompson Sampling acquisition function.
    
    Draws a single sample from the posterior distribution (mu + sigma * N(0,1))
    and selects the point with the minimum sampled value divided by its cost.
    This stochastic approach naturally balances exploration and exploitation
    without tuning a beta parameter.
    
    Parameters:
    -----------
    mu : array_like, shape (2000,)
        GP mean predictions.
    sigma : array_like, shape (2000,)
        GP standard deviation predictions.
    f_best : float
        Best observed function value so far.
    cost : array_like, shape (2000,)
        Cost of evaluating each point (1 + mean(x) for x in [0,1]^6).
    
    Returns:
    --------
    scores : array_like, shape (2000,)
        Acquisition values; higher is better.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is non-negative
    sigma_safe = np.maximum(sigma, 0.0)
    
    # Draw a single sample from the posterior: mu + sigma * N(0,1)
    rng = np.random.default_rng()
    noise = rng.standard_normal(len(mu))
    samples = mu + sigma_safe * noise
    
    # Score: negative of (sampled value / cost) so that lower sampled value / cost is better
    # Since we want higher scores to be better, we negate the ratio
    scores = -samples / cost
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
