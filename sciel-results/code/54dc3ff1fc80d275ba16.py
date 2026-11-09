"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    # Lower Confidence Bound (LCB) for minimization.
    # We want to minimize mu - beta * sigma.
    # The evaluator selects the candidate with the HIGHEST score.
    # So we return score = beta * sigma - mu.
    # A beta of 2.5 is a common heuristic for global optimization
    # of multimodal functions like Branin, balancing exploration
    # and exploitation.
    
    # Ensure sigma is non-negative and finite
    sigma = np.maximum(sigma, 1e-9)
    
    beta = 2.5
    scores = beta * sigma - mu
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
