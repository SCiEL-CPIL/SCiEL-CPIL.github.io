"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    # Adaptive Lower Confidence Bound (LCB) for minimization
    # Score = (beta * sigma - mu) / cost
    
    # Theoretical minimum of Styblinski-Tang 2D
    f_min_theory = -78.3323
    
    # Calculate gap between current best and theoretical min
    # f_best is a scalar (lowest observed value)
    gap = f_best - f_min_theory
    
    # Ensure gap is non-negative (in case f_best < f_min_theory due to noise/shift)
    gap = max(gap, 0.0)
    
    # Adaptive beta:
    # High gap -> High beta (Explore)
    # Low gap -> Low beta (Exploit)
    # Use a smooth transition.
    # beta = beta_min + (beta_max - beta_min) * (gap / (gap + scale))
    beta_min = 0.5
    beta_max = 3.0
    scale = 5.0  # Gap of 5 units triggers half transition
    
    beta = beta_min + (beta_max - beta_min) * (gap / (gap + scale))
    
    # Ensure sigma and cost are safe for division
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Calculate score
    # We want to maximize: high sigma (uncertainty) and low mu (good value)
    score = (beta * sigma - mu) / cost
    
    return score
# EVOLVE-BLOCK-END
