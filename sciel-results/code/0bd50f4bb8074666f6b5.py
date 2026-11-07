"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Module-level cache to avoid recomputing constants
_START_TIME = time.time()
_DEADLINE = _START_TIME + 325.0  # safety margin before 330s

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a hybrid of Expected Improvement (EI)
    and a local improvement bonus, scaled by cost.
    
    Parameters:
    -----------
    mu : ndarray, shape (N,)
        Mean predictions from the GP.
    sigma : ndarray, shape (N,)
        Standard deviation predictions from the GP.
    f_best : float
        Best observed value so far.
    cost : ndarray, shape (N,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : ndarray, shape (N,)
        Acquisition scores (higher = more desirable).
    """
    # Deadline guard
    if time.time() > _DEADLINE:
        # If we're close to timeout, just return a simple score to avoid hanging
        return np.zeros_like(mu)
    
    # Ensure sigma is positive for numerical stability
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute z-score for Expected Improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (EI)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost-aware scaling: divide by cost to penalize expensive points
    # But we also want to ensure that very cheap points with moderate EI are favored
    # The cost is in [1, 2] since it's 1 + mean(x_scaled) and x_scaled is in [0,1]
    # So cost is between 1 and 2
    
    # Hybrid: combine EI/cost with a small exploration term based on uncertainty
    # This helps in early stages when the GP is uncertain
    # Use a weighted combination
    ei_per_cost = ei / cost
    
    # Add a small bonus for high uncertainty (exploration) but scaled down
    # to not dominate the EI signal
    exploration_bonus = 0.1 * sigma_safe / cost
    
    scores = ei_per_cost + exploration_bonus
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
