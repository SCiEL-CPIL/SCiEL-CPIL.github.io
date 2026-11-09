# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition: UCB with adaptive beta, divided by cost.
    Uses a robust UCB formulation that balances exploration (via sigma)
    and exploitation (via mu), scaled by the inverse cost to account
    for the variable evaluation cost.
    """
    # Clip sigma to avoid division by zero and numerical issues
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Adaptive beta: start with a moderate exploration constant
    # We can use a fixed beta or make it adaptive. For robustness, 
    # let's use a standard UCB with a reasonable beta.
    # Since we don't have the number of iterations directly, we use a 
    # heuristic beta. A common choice is sqrt(2 * log(N)) but N is 
    # not passed. We'll use a fixed beta that works well in practice.
    # Let's try beta = 1.5 as a starting point, which is a common 
    # value for UCB in BO.
    beta = 1.5
    
    # UCB: mu + beta * sigma
    # We want to maximize this, but also account for cost.
    # The objective is to minimize, so lower mu is better.
    # UCB for minimization: mu - beta * sigma (we want to explore low values)
    # Wait, for minimization, UCB is typically mu - beta * sigma.
    # Let's use that.
    ucb = mu - beta * sigma_safe
    
    # Divide by cost to get cost-aware score
    # Higher score = better
    score = ucb / cost
    
    # Ensure finite values
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
# EVOLVE-BLOCK-END
