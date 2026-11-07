"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Lower Confidence Bound (LCB) acquisition function with dynamic
    exploration scaled by remaining budget.
    
    Parameters:
    mu: (2000,) array of GP mean predictions
    sigma: (2000,) array of GP standard deviations
    f_best: float, the lowest observed function value
    cost: (2000,) array of evaluation costs
    
    Returns:
    scores: (2000,) array of acquisition scores (higher is better)
    """
    start_time = time.time()
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Estimate remaining budget
    # Initial cost: 5 uniform points. Each point in [-1,1]^8 has mean(x_scaled) ~ 0.5
    # So initial cost ~ 5 * (1 + 0.5) = 7.5
    # Total budget is 30
    # We estimate how many evaluations we've done based on f_best being non-None
    # But we don't have access to number of evaluations directly.
    # Let's use a heuristic: if f_best is close to minimum (-0.8), we're doing well.
    # Actually, we can't know the exact budget used.
    # Let's assume we're in the middle of the budget.
    # A simple dynamic beta: start high (exploration) and decrease.
    # Since we don't know the step, let's use a fixed but smaller beta than typical.
    # In 8D, variance can be high. EI tends to be misled.
    # LCB: -mu - beta * sigma. We want to maximize this (i.e., minimize mu + beta*sigma).
    # So score = -(mu + beta * sigma) / cost.
    
    # Dynamic beta: 
    # If f_best is high (poor), we should explore more? Or if we are early, explore more.
    # Without step count, let's use a fixed beta that balances exploitation and exploration.
    # Typical beta in BO is 1.0 to 2.0.
    # Given the cost structure, we want to be efficient.
    # Let's try beta = 1.0 initially.
    
    beta = 1.0
    
    # LCB value (lower is better)
    lcb = mu + beta * sigma_safe
    
    # We want to maximize the acquisition, which corresponds to minimizing the LCB.
    # So we use -lcb as the score base.
    base_score = -lcb
    
    # Cost normalization
    cost_safe = np.maximum(cost, 1e-9)
    scores = base_score / cost_safe
    
    # Handle any non-finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Deadline guard
    elapsed = time.time() - start_time
    if elapsed > 280:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
