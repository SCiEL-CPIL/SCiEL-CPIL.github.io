import numpy as np
import time

# Global start time for deadline checking
_START_TIME = time.time()
# Deadline: 330s total wall-clock for the evaluator.
# We should return before 325s to be safe.
DEADLINE = 325.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Cost-Effective UCB (CE-UCB).
    
    Args:
        mu: Mean of the GP (standardized objective units), shape (2000,)
        sigma: Standard deviation of the GP (standardized objective units), shape (2000,)
        f_best: Best observed value (standardized objective units), scalar
        cost: Cost of evaluating each candidate, shape (2000,)
        
    Returns:
        scores: Acquisition scores, shape (2000,), higher is better
    """
    # Check deadline
    if time.time() > DEADLINE:
        # Return a safe default (e.g., based on sigma only, or zeros)
        # If we are out of time, we should return something valid quickly.
        # Returning sigma/cost encourages exploration of uncertain cheap areas.
        return sigma / np.maximum(cost, 1e-9)

    # Ensure inputs are numpy arrays
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    cost = np.asarray(cost)
    
    # Safety: clip sigma to avoid division by zero or numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Safety: ensure cost is positive
    cost_safe = np.maximum(cost, 1e-9)
    
    # Cost-Effective UCB for minimization
    # We want to maximize: (-mu + beta * sigma) / cost
    # beta controls the exploration-exploitation trade-off
    beta = 2.0
    
    scores = (-mu + beta * sigma_safe) / cost_safe
    
    # Ensure finiteness
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
