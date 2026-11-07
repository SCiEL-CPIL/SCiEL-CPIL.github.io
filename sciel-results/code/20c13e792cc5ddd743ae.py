import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 320.0  # Safety margin before 330s limit

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Lower Confidence Bound (LCB) acquisition function.
    
    The LCB acquisition function balances exploitation (low predicted mean) 
    and exploration (high predicted uncertainty) by minimizing:
        LCB(x) = mu(x) - beta * sigma(x)
    
    Since the evaluator selects the point with the highest score, we return:
        Score(x) = -LCB(x) / Cost(x)
    
    This formulation directly targets points that are likely to have low function values 
    (good LCB) and low evaluation costs (high score due to division by cost).
    """
    # Deadline guard
    if time.time() > _DEADLINE:
        # Fallback to a simple cost-aware mean minimization
        cost_safe = np.where(np.isfinite(cost) & (cost > 1e-6), cost, 1.0)
        mu_safe = np.where(np.isfinite(mu), mu, f_best)
        return -mu_safe / cost_safe

    # Sanitize inputs
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    sigma = np.maximum(sigma, 1e-6)
    mu = np.where(np.isfinite(mu), mu, f_best)
    cost = np.where(np.isfinite(cost) & (cost > 1e-6), cost, 1.0)
    
    n = len(mu)
    if n == 0:
        return np.array([])
        
    # Exploration parameter for LCB
    # A moderate value is chosen to balance exploitation and exploration.
    # Given the cost structure (center is cheap), we can afford to be slightly 
    # more exploratory to find the true basin.
    beta = 1.5
    
    # Calculate LCB: mu - beta * sigma
    # We want to minimize LCB, so we maximize -LCB
    # -LCB = beta * sigma - mu
    lcb_score = beta * sigma - mu
    
    # Cost-aware score: higher is better
    # Points with low cost (small denominator) get higher scores
    score = lcb_score / cost
    
    # Handle potential numerical issues
    score = np.where(np.isfinite(score), score, -1e6)
    
    # Normalize scores to [0, 1]
    # This ensures the output is in a consistent range, though argmax is invariant to affine transformations.
    score_min = np.min(score)
    score_max = np.max(score)
    
    if score_max > score_min:
        score = (score - score_min) / (score_max - score_min)
    else:
        # If all scores are identical, return uniform or zero
        score = np.zeros_like(score)
        
    return score
