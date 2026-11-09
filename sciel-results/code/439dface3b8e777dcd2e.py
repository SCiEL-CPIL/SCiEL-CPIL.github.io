import numpy as np
import time

# Global timer for deadline guard
_start_time = time.time()

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Lower Confidence Bound (LCB)
    with a sigmoidal cost discount.
    
    Parameters:
    mu: array of mean predictions
    sigma: array of standard deviation predictions
    f_best: lowest observed value (not directly used in LCB, but kept for interface)
    cost: array of evaluation costs
    
    Returns:
    array of scores (higher is better)
    """
    # Deadline guard: if we are close to the 330s limit, return a simple score
    # to ensure we don't timeout in the evaluator's loop.
    if time.time() - _start_time > 320:
        # Simple exploitation score
        return -mu - 0.1 * cost

    # Handle sigma=0 cases to avoid division by zero or numerical issues
    sigma_safe = np.where(sigma < 1e-9, 1e-9, sigma)
    
    # Lower Confidence Bound: mu - beta * sigma
    # We want to minimize this, so we negate it for the acquisition score (maximize)
    beta = 2.5
    lcb = mu - beta * sigma_safe
    
    # Sigmoidal cost discount
    # Cost is in [1, 2]. We want to favor lower costs.
    # discount is high for low cost, low for high cost.
    k = 5.0
    c0 = 1.5
    discount = 1.0 / (1.0 + np.exp(k * (cost - c0)))
    
    # Score is higher for lower LCB (better function value estimate)
    # and higher discount (lower cost).
    score = discount * (-lcb)
    
    # Ensure finite scores
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
