"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Lower Confidence Bound (LCB).
    LCB = mu - beta * sigma.
    We maximize -LCB / cost to select the next point.
    """
    beta = 2.0
    # Calculate LCB
    lcb = mu - beta * sigma
    
    # We want to minimize LCB, so we maximize -LCB.
    # Divide by cost to account for cost-awareness.
    score = -lcb / cost
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
