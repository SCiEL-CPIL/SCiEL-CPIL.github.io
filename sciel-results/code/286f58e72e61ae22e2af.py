"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    # Cost-aware Upper Confidence Bound (UCB) acquisition function.
    # We are minimizing the objective, so we prefer candidates with low predicted mean (mu)
    # and high uncertainty (sigma).
    # The score is adjusted by the evaluation cost (higher cost -> lower score).
    
    # Beta is the exploration parameter.
    # A value of 3.0 encourages sufficient exploration to navigate the multimodal
    # landscape of Hartmann 3D, especially given the small initial budget.
    beta = 3.0
    
    # Ensure cost is safe for division
    safe_cost = np.where(cost > 1e-8, cost, 1.0)
    
    # Calculate UCB score
    # -mu because we want to minimize
    # + beta * sigma for exploration
    scores = (-mu + beta * sigma) / safe_cost
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
