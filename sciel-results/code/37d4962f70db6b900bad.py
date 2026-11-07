"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global state for deadline tracking
_start_time = None
_DEADLINE = 325.0  # Safety margin before 330s kill


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Args:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP standard deviation predictions
        f_best: float - best observed value so far
        cost: array of shape (2000,) - evaluation costs (1 + mean(x scaled to [0,1]))
    
    Returns:
        array of shape (2000,) - acquisition scores (higher = evaluate next)
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        # Return a safe fallback: just use -mu/cost to avoid any further computation
        return -mu / (cost + 1e-9)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-8)
    
    # Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z), where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Cost-aware: divide EI by cost
    # Cost is 1 + mean(x_scaled), so ranges from 1 to 2
    ei_per_cost = ei / cost
    
    # Add a small exploration bonus based on uncertainty
    # Use a weighted combination: primarily EI/cost with a small UCB-like term
    # The UCB term helps explore regions with high uncertainty
    ucb_term = -mu + 1.5 * sigma  # beta=1.5 for moderate exploration
    
    # Normalize the UCB term to be on a similar scale to EI/cost
    # EI/cost is in objective units per cost unit
    # ucb_term is in objective units
    # Scale ucb_term by a small factor to keep it as a secondary signal
    score = ei_per_cost + 0.05 * ucb_term
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
# EVOLVE-BLOCK-END
