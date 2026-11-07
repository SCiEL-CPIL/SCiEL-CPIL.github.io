"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Lower Confidence Bound (LCB) acquisition function.
    
    Args:
        mu: Array of shape (N,) with mean predictions.
        sigma: Array of shape (N,) with standard deviations.
        f_best: Scalar, the best observed function value so far.
        cost: Array of shape (N,) with evaluation costs.
        
    Returns:
        Array of shape (N,) with scores (higher is better).
    """
    # Normalize objective values to be scale-invariant
    # f_best is the lowest observed. 
    # If f_best is close to 0, scale is 1.
    # If f_best is large, scale matches the magnitude.
    # This helps if different functions have different scales.
    scale = max(1.0, abs(float(f_best)))
    
    mu_norm = mu / scale
    sigma_norm = sigma / scale
    
    # LCB for minimization: mu - kappa * sigma
    # We want to minimize LCB.
    # Score should be higher for lower LCB.
    # So Score = -LCB = -mu + kappa * sigma
    
    # Exploration parameter
    kappa = 2.0
    
    # Cost aversion parameter
    # Cost is in [1, 2].
    # We subtract lambda * cost.
    # If lambda is too high, we always pick the cheapest point.
    # If lambda is too low, we ignore cost.
    # With normalized mu/sigma (O(1)), a lambda of 0.5 gives cost a weight
    # comparable to the objective uncertainty.
    lambda_cost = 0.5
    
    # Calculate Score
    score = -mu_norm + kappa * sigma_norm - lambda_cost * cost
    
    # Ensure finite values
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
# EVOLVE-BLOCK-END
