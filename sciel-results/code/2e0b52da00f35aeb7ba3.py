# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard: track when the evaluation started.
# The evaluator kills after 330s wall-clock. We set a conservative deadline.
# We don't know when the process started, but we can track relative time within our function calls.
# However, the function is called many times. We need a global deadline.
# The problem says "Each evaluation must finish within 300 s". This likely refers to the entire BO loop.
# Since we can't know the start time of the loop, we assume the first call is near the start.
# A safer approach: assume the total budget is 300s from the first call.
_start_time = None

def _get_remaining_time():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    elapsed = time.time() - _start_time
    # Leave a safety margin of 5 seconds
    return max(0.0, 300.0 - elapsed - 5.0)

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
    mu: array of shape (2000,) - GP mean
    sigma: array of shape (2000,) - GP standard deviation
    f_best: float - best observed value
    cost: array of shape (2000,) - cost of evaluating each point
    
    Returns:
    array of shape (2000,) - acquisition scores (higher is better)
    """
    # Check time remaining
    remaining = _get_remaining_time()
    if remaining < 1.0:
        # If we're running out of time, return a simple fallback
        # Prioritize low cost and low mu
        return (f_best - mu) / (cost + 1e-9)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Compute z-score
    z = (f_best - mu) / sigma
    
    # Expected Improvement (EI)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Cost-aware acquisition: EI per unit cost
    # We can also add a small exploration term based on sigma
    # But let's stick with a robust cost-aware EI
    
    # To handle potential numerical issues, clip the scores
    scores = ei / (cost + 1e-9)
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
