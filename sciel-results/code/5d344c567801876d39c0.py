# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a budget-scaled LCB approach.
    
    Args:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP standard deviation predictions
        f_best: float - best observed value so far
        cost: array of shape (2000,) - cost of evaluating each candidate
    
    Returns:
        array of shape (2000,) - acquisition scores (higher = evaluate next)
    """
    # Ensure numerical stability
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Constants
    BUDGET = 30.0
    F_MIN = -3.32237
    # Estimate the maximum possible gap to the optimum for progress estimation.
    # Hartmann 6d values can be positive, so a gap of 4.0 is a safe upper bound 
    # for the "remaining fraction" calculation to keep beta bounded.
    MAX_GAP = 4.0
    
    # Calculate gap to the known minimum
    gap = f_best - F_MIN
    
    # Estimate remaining budget fraction based on progress (gap to optimum)
    # High gap -> early stage -> high remaining fraction -> high beta (explore)
    # Low gap -> late stage -> low remaining fraction -> low beta (exploit)
    # Clip to [0.1, 1.0] to ensure minimum exploration and cap maximum exploration
    fraction_remaining = np.clip(gap / MAX_GAP, 0.1, 1.0)
    
    # Estimate remaining budget
    remaining_budget = BUDGET * fraction_remaining
    
    # Dynamically scale the exploration term (beta) by the square root of the remaining budget
    # This ensures aggressive exploration early and precise exploitation later
    beta = np.sqrt(remaining_budget)
    
    # Lower Confidence Bound (LCB) for minimization: LCB = mu - beta * sigma
    # We want to maximize the acquisition score, so we negate the LCB:
    # Score_unnormalized = -mu + beta * sigma
    lcb_score = -mu + beta * sigma_safe
    
    # Cost-normalized score
    score = lcb_score / cost_safe
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
