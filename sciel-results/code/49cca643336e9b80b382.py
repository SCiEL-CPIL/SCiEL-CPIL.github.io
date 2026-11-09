"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Lower Confidence Bound (LCB) / UCB for minimization.
    
    We use the UCB approach:
    1. Compute the Lower Confidence Bound: LCB = mu - sqrt(beta) * sigma.
       This balances exploitation (low mu) and exploration (high sigma).
    2. We want to minimize LCB, so we maximize the score: Score = -LCB.
    3. Divide by cost to account for the cost-aware requirement.
    
    This approach is distinct from Expected Improvement (EI) as it uses a linear
    combination of mean and standard deviation with a tunable exploration parameter beta.
    """
    # Ensure sigma is positive and not too small
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Tuning parameter for exploration/exploitation trade-off
    # beta controls the weight of the uncertainty term.
    # A value of 1.0 is a reasonable starting point for 2D problems.
    beta = 1.0
    
    # Compute LCB (for minimization, we look at mu - sqrt(beta)*sigma)
    # We want to select points with LOW LCB.
    lcb = mu - np.sqrt(beta) * sigma_safe
    
    # Convert to a maximization score: higher score means lower LCB (better)
    # We negate the LCB.
    score = -lcb
    
    # Cost-aware: divide by cost
    # Ensure cost is not zero or too small
    cost_safe = np.maximum(cost, 1e-8)
    
    result = score / cost_safe
    
    # Ensure all values are finite
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
# EVOLVE-BLOCK-END
