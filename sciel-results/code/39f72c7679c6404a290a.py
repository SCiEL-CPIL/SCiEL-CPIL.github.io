"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Upper Confidence Bound (UCB) for minimization.
    # We want to maximize the potential for low values, which is mu - kappa * sigma.
    # So we score by -(mu - kappa * sigma) = kappa * sigma - mu.
    # This is a structural change from EI, offering a different exploration/exploitation balance.
    kappa = 2.5
    ucb_score = kappa * sigma - mu
    
    # Normalize by cost to account for evaluation expense.
    # Add a small epsilon to cost to avoid division by zero, though cost should be > 0.
    return ucb_score / (cost + 1e-9)
# EVOLVE-BLOCK-END
