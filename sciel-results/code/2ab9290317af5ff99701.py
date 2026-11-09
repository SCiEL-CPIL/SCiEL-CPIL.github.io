"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function.
    
    Uses a Cost-Adjusted Lower Confidence Bound (LCB) approach.
    The exploration term is scaled by the inverse square root of the cost,
    treating expensive evaluations as having higher effective variance in the
    decision process, but penalizing exploration less severely than linear
    cost division might.
    
    Parameters:
    mu: (2000,) array of GP means.
    sigma: (2000,) array of GP standard deviations.
    f_best: float, best observed value.
    cost: (2000,) array of evaluation costs.
    
    Returns:
    (2000,) array of scores. Higher is better.
    """
    # Ensure inputs are arrays
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    cost = np.asarray(cost)
    
    # Safety margin for cost to avoid division by zero or very small costs
    # Cost is 1 + mean(x_scaled), so min cost is 1.0.
    # But we add a tiny epsilon for numerical stability.
    safe_cost = cost + 1e-9
    
    # Cost adjustment factor: 1 / sqrt(cost)
    # This penalizes exploration in high-cost regions, but less than 1/cost.
    cost_factor = 1.0 / np.sqrt(safe_cost)
    
    # Exploration coefficient beta
    # A typical value for LCB in BO is around 1.5 to 3.0.
    # We use 2.0 as a balanced default.
    beta = 2.0
    
    # LCB for minimization is mu + beta * sigma.
    # We want to MINIMIZE this, so we MAXIMIZE the negative.
    # Acquisition = - (mu + beta * sigma * cost_factor)
    # Note: The cost factor applies to the exploration term (sigma).
    # Does it apply to the exploitation term (mu)?
    # The directive says: "exploration term is scaled by the inverse square root of the cost".
    # So, A(x) = -mu(x) - beta * sigma(x) * (1/sqrt(cost(x)))
    
    acq = -mu - beta * sigma * cost_factor
    
    # Ensure finite scores
    acq = np.nan_to_num(acq, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return acq
# EVOLVE-BLOCK-END
