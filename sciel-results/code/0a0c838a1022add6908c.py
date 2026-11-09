import numpy as np
import time

_start_time = time.time()
_TIMEOUT_LIMIT = 290.0  # Safety margin for 300s limit

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Cost-Adjusted LCB.
    
    Parameters:
    mu: array of shape (N,), GP mean predictions
    sigma: array of shape (N,), GP standard deviation predictions
    f_best: float, best observed value so far
    cost: array of shape (N,), cost of evaluating each candidate
    
    Returns:
    scores: array of shape (N,), higher is better
    """
    # Deadline guard
    if time.time() - _start_time > _TIMEOUT_LIMIT:
        # Fallback: Greedy selection based on mean, penalized by cost
        # We want to minimize mu, so score = -mu / cost
        return -mu / cost

    # LCB parameter (exploration/exploitation trade-off)
    kappa = 3.0
    
    # Calculate Lower Confidence Bound
    # LCB(x) = mu(x) - kappa * sigma(x)
    # We want to minimize LCB, so we maximize -LCB
    lcb = mu - kappa * sigma
    
    # Cost-adjusted score
    # Divide by cost to favor cheaper evaluations
    # Ensure cost is positive (it is, >= 1)
    scores = -lcb / cost
    
    # Ensure finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
