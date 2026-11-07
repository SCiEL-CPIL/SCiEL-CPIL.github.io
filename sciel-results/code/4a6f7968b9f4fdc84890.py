"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    # Cost-aware Lower Confidence Bound (LCB)
    # This encourages exploration in uncertain regions (high sigma)
    # and exploitation in promising regions (low mu), balanced by cost.
    
    beta = 2.5
    lcb = mu - beta * sigma
    improvement = f_best - lcb
    
    # Ensure scores are non-negative
    improvement = np.maximum(improvement, 0.0)
    
    # Avoid division by zero
    cost_safe = np.maximum(cost, 1e-6)
    
    return improvement / cost_safe
# EVOLVE-BLOCK-END
