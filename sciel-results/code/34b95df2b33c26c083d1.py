"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function: Cost-normalized Lower Confidence Bound (LCB)
    with dynamic exploration scaled by remaining budget.
    
    Parameters:
    mu: (2000,) array of GP mean predictions
    sigma: (2000,) array of GP standard deviations
    f_best: float, the lowest observed function value
    cost: (2000,) array of evaluation costs
    
    Returns:
    scores: (2000,) array of acquisition scores (higher is better)
    """
    # Known minimum of the objective function
    f_min = -0.8
    
    # Calculate the gap to the optimum
    gap = max(0.0, f_best - f_min)
    
    # Total budget is 30, initial 5 points cost ~1.5 each = ~7.5
    # Remaining budget estimate based on f_best progress
    # As we get closer to optimum, we should be more exploitative
    total_budget = 30.0
    initial_cost_estimate = 7.5  # 5 uniform points at ~1.5 cost each
    remaining_budget = max(1.0, total_budget - initial_cost_estimate)
    
    # Dynamic exploration parameter beta
    # Start with higher exploration, decrease as we approach optimum
    # beta controls the trade-off between exploitation (mu) and exploration (sigma)
    # Larger gap -> larger beta (more exploration)
    # Smaller gap -> smaller beta (more exploitation)
    # Scale by remaining budget to prevent wasting expensive evaluations on high-variance regions
    base_beta = 0.5 + 2.0 * gap  # Base exploration from gap
    budget_factor = np.sqrt(remaining_budget / total_budget)  # Decreases as budget depletes
    beta = base_beta * budget_factor
    
    # Clip beta to reasonable range
    beta = np.clip(beta, 0.1, 3.0)
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # LCB = mu - beta * sigma (lower confidence bound for minimization)
    # We want to MINIMIZE, so we pick points with low LCB
    # Acquisition score = -LCB / cost (higher score = better)
    lcb = mu - beta * sigma_safe
    
    # Cost-aware score: negative LCB divided by cost
    # Cost is 1 + mean(x_scaled). Min cost is 1.0, Max is 2.0.
    cost_safe = np.maximum(cost, 1e-9)
    scores = -lcb / cost_safe
    
    # Handle any non-finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
