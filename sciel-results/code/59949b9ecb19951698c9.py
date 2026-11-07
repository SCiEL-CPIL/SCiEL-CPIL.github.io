"""Rewritten program with adaptive UCB acquisition function."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state for deadline guard
_start_time = time.time()
_TIME_LIMIT = 300.0  # Safety margin before the 330s hard limit

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using adaptive Upper Confidence Bound (UCB).
    
    Parameters:
    mu (np.ndarray): GP mean predictions, shape (2000,)
    sigma (np.ndarray): GP standard deviations, shape (2000,)
    f_best (float): Lowest observed function value
    cost (np.ndarray): Cost of evaluating each candidate, shape (2000,)
    
    Returns:
    np.ndarray: Scores for 2000 candidates (higher = evaluate next)
    """
    # Deadline guard: if we are close to the time limit, return neutral scores
    # to avoid any further computation overhead or risk of timeout in the evaluator loop.
    if time.time() - _start_time > _TIME_LIMIT:
        return np.zeros_like(mu)

    # Ensure inputs are numpy arrays for vectorized operations
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    cost = np.asarray(cost)

    # Avoid division by zero or negative costs (though cost should be >= 1)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Avoid division by zero in sigma if needed, though UCB handles sigma=0 gracefully
    # (it just becomes exploitation).
    
    # Adaptive Kappa
    # We want high exploration (high kappa) when f_best is poor (large value).
    # We want low exploration (low kappa) when f_best is good (small value, close to 0).
    # Using log1p to handle large f_best values gracefully.
    # f_best is in objective units.
    kappa = 1.0 + 0.5 * np.log1p(f_best)
    
    # UCB Score for Minimization
    # We want to maximize this score.
    # Low mu is good -> -mu is high.
    # High sigma is good -> +kappa * sigma is high.
    # Low cost is good -> divide by cost.
    scores = (-mu + kappa * sigma) / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
