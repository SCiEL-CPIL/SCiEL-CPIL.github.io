import numpy as np
import time

# Global start time to act as a safety net if the evaluator keeps the process alive
_START_TIME = time.time()
_DEADLINE = 330.0
_SAFETY_MARGIN = 10.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Lower Confidence Bound (LCB).
    
    We maximize score = (beta * sigma - mu) / cost.
    This is equivalent to minimizing (mu - beta * sigma) * cost? No, it's minimizing mu - beta*sigma while penalizing high cost.
    Actually, maximizing (beta*sigma - mu)/cost is a specific cost-aware formulation.
    
    Args:
        mu: (2000,) array of GP means.
        sigma: (2000,) array of GP standard deviations.
        f_best: float, the lowest observed value.
        cost: (2000,) array of costs for each candidate.
        
    Returns:
        (2000,) array of scores (higher is better).
    """
    # Check time limit
    if time.time() > _DEADLINE - _SAFETY_MARGIN:
        return np.zeros_like(mu)

    # Ensure sigma is non-negative and safe for operations
    sigma = np.maximum(sigma, 1e-9)
    
    # Ensure cost is positive and safe for division
    cost = np.maximum(cost, 1e-9)
    
    # Beta parameter for exploration
    # For Rastrigin, a higher beta encourages jumping to new basins.
    # 2.0 is a common aggressive choice.
    beta = 2.0
    
    # LCB score to maximize: beta * sigma - mu
    # We divide by cost to make it cost-aware
    lcb_score = (beta * sigma - mu) / cost
    
    # Handle numerical issues
    lcb_score = np.nan_to_num(lcb_score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return lcb_score
