"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Lower Confidence Bound (LCB) per unit cost.
    # LCB = mu - beta * sigma
    # We want to minimize LCB, so we maximize -LCB / cost.
    # Score = (-mu + beta * sigma) / cost
    
    beta = 2.0
    lcb = mu - beta * sigma
    scores = -lcb / (cost + 1e-9)
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
