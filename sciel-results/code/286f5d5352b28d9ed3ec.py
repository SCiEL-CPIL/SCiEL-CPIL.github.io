"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        Mean of the GP posterior at 2000 candidate points.
    sigma : array of shape (2000,)
        Standard deviation of the GP posterior at 2000 candidate points.
    f_best : float
        Best (lowest) observed function value so far.
    cost : array of shape (2000,)
        Cost of evaluating each candidate (1 + mean of coordinates scaled to [0,1]).
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    # Deadline guard: track elapsed time to avoid exceeding 300s per evaluation
    start_time = time.time()
    
    # Ensure sigma is positive for numerical stability
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement (EI)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Add a small exploration bonus weighted by sigma to encourage early exploration
    # This helps escape local minima and explore the space more effectively
    exploration_bonus = 0.05 * sigma_safe
    
    # Combine EI and exploration bonus
    raw_score = ei + exploration_bonus
    
    # Normalize by cost to get cost-aware acquisition
    # cost is already 1 + mean(x scaled to [0, 1]), so it's in [1, 2]
    cost_safe = np.maximum(cost, 1e-9)
    scores = raw_score / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check deadline: if we're getting close to 300s, return immediately
    elapsed = time.time() - start_time
    if elapsed > 280:  # Safety margin
        return scores
    
    return scores
# EVOLVE-BLOCK-END
