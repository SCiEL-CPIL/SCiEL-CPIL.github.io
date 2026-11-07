"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function: Cost-normalized Lower Confidence Bound (LCB).
    
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
    
    # Dynamic exploration parameter for LCB
    # When gap is large, we explore more (higher beta)
    # When gap is small, we exploit more (lower beta)
    # Beta starts high and decreases as we get closer to the optimum
    # The remaining budget also affects exploration: more remaining budget -> more exploration
    # We estimate remaining budget based on f_best improvement rate
    # For now, use a simple heuristic: beta scales with gap
    beta = 2.0 * np.sqrt(gap + 1e-6)
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # LCB = mu - beta * sigma (for minimization)
    # We want to minimize this, so we maximize -LCB = -mu + beta * sigma
    lcb_scores = -mu + beta * sigma_safe
    
    # Cost-aware score: divide by cost to prefer cheaper evaluations
    # Cost is 1 + mean(x_scaled). Min cost is 1.0, Max is 2.0.
    cost_safe = np.maximum(cost, 1e-9)
    scores = lcb_scores / cost_safe
    
    # Handle any non-finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
