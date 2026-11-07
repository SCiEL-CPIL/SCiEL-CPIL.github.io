"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    # Lower Confidence Bound (LCB) for minimization
    # We minimize mu - kappa * sigma, so we maximize -mu + kappa * sigma
    # kappa=1 was identified as an effective balance for exploration in Ackley 2D
    kappa = 1.0
    scores = -mu + kappa * sigma
    
    # Ensure all scores are finite to satisfy the contract
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
