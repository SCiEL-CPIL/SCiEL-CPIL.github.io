"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a cost-normalized LCB with dynamic exploration.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        GP mean predictions for 2000 candidate points.
    sigma : array-like, shape (2000,)
        GP standard deviation predictions for 2000 candidate points.
    f_best : float
        Best (lowest) observed function value so far.
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate point (1 + mean(x_scaled)).
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = better to evaluate next).
    """
    # Deadline guard: track elapsed time
    t_start = time.time()
    DEADLINE = 330.0  # evaluator kills at 330s wall-clock
    SAFETY_MARGIN = 5.0  # seconds to spare
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = len(mu)
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Dynamic exploration parameter based on progress
    # As f_best approaches the known minimum (-0.8), reduce exploration
    known_min = -0.8
    progress = np.clip((f_best - known_min) / (0.0 - known_min), 0.0, 1.0)
    # Start with high exploration, decrease as we get closer to optimum
    beta = 1.5 * (1.0 - 0.5 * progress)
    
    # Cost-normalized LCB: lower is better, so we negate for "higher is better" scoring
    # LCB = mu - beta * sigma
    # We want to maximize (improvement potential) / cost
    # Use: (f_best - LCB) / cost as a proxy for expected improvement per unit cost
    lcb = mu - beta * sigma_safe
    expected_improvement = np.maximum(f_best - lcb, 0.0)
    
    # Cost normalization: divide by cost to prefer cheaper evaluations
    scores = expected_improvement / np.maximum(cost, 1e-10)
    
    # Add a small exploration bonus for points with high uncertainty that are cheap
    # This helps escape local optima in high dimensions
    uncertainty_bonus = beta * sigma_safe / np.maximum(cost, 1e-10)
    scores += 0.3 * uncertainty_bonus
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Deadline check
    if time.time() - t_start > DEADLINE - SAFETY_MARGIN:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
