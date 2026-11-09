# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Module-level constants for time management
_START_TIME = time.time()
_DEADLINE = _START_TIME + 320.0  # Safety margin before 330s kill

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function based on Cost-Adjusted Lower Confidence Bound (LCB).
    
    Structural Change: Replaces Expected Improvement (EI) with LCB.
    LCB directly targets the global minimum by maximizing -(mu + kappa * sigma) / cost.
    This shifts the objective from incremental regret reduction (EI) to finding the 
    lowest possible value per unit budget.
    
    Parameters
    ----------
    mu     : (N,) array of GP mean predictions
    sigma  : (N,) array of GP standard deviations
    f_best : float, the best (lowest) observed value so far
    cost   : (N,) array of evaluation costs for each candidate

    Returns
    -------
    scores : (N,) array of finite scores, higher = evaluate next
    """
    # Deadline guard
    if time.time() > _DEADLINE:
        # Fallback: return random finite scores to avoid invalidity
        return np.random.rand(len(mu))
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    # Ensure cost is positive
    cost_safe = np.maximum(cost, 1e-9)
    
    f_best = float(f_best)
    
    # Dynamic kappa based on remaining budget ratio.
    # Total budget is 30. Initial cost is ~5 (5 points * avg cost 1.5).
    # As cost accumulates, remaining_budget decreases.
    # We approximate the current total cost using the mean of the current cost array 
    # scaled by the number of points? No, cost array is for 2000 candidates.
    # We don't know the exact number of evaluations made. 
    # However, the prompt says "While total cost is below 30".
    # A robust heuristic for "remaining budget" without explicit state:
    # If we assume the process is in the middle, we can use a fixed exploration schedule 
    # or infer from the spread of sigma?
    # Actually, simpler: use a fixed kappa that balances exploration and exploitation well.
    # The inspiration suggests dynamic scaling, but without knowing `n_evals` or `total_cost` 
    # explicitly (only the current candidate costs), we can't compute exact remaining budget.
    # However, we can use the fact that f_best is available.
    # Let's stick to a strong, fixed kappa for LCB which is known to be robust in BO.
    # A common value is kappa ~ 2-5.
    # Let's try kappa = 2.0.
    
    kappa = 2.0
    
    # Compute LCB: mu + kappa * sigma
    lcb = mu + kappa * sigma_safe
    
    # Cost-aware scoring: Maximize -LCB / cost
    # This penalizes high cost and high predicted value (low LCB)
    # We want to find the point with the lowest predicted value adjusted for uncertainty, 
    # relative to its cost.
    
    # Using division by cost directly can be unstable if cost varies a lot.
    # The inspiration 1 used: ei / (1 + 0.5 * (cost - 1.0))
    # Let's use a similar soft penalty for cost in the denominator.
    # cost_penalty = 1.0 + 0.5 * (cost_safe - 1.0)
    
    # Alternative: simple division -LCB / cost_safe
    # This is aggressive. Let's try the soft penalty version which was in a higher scoring inspiration (1.2547).
    cost_penalty = 1.0 + 0.5 * (cost_safe - 1.0)
    
    # Score is negative LCB divided by cost penalty
    # Since we want to MAXIMIZE the score, and LCB is the lower bound (smaller is better),
    # we take -LCB.
    scores = -lcb / cost_penalty
    
    # Ensure all scores are finite
    if not np.all(np.isfinite(scores)):
        scores = np.where(np.isfinite(scores), scores, -1e9)
        
    return scores
# EVOLVE-BLOCK-END
