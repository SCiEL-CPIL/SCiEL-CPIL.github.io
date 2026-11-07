"""Rewritten program using Upper Confidence Bound (UCB) acquisition."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Use Upper Confidence Bound (UCB) for exploration.
    # Since we want to minimize, we maximize (f_best - mu) + kappa * sigma.
    # But standard UCB for minimization is: minimize mu - kappa * sigma.
    # To fit the "higher score = better" interface, we return:
    # score = -mu + kappa * sigma
    # where kappa is the exploration parameter.
    
    # A common choice for kappa is time-dependent or fixed.
    # For 30 evaluations total, with 5 initial, we have 25 optimization steps.
    # A fixed kappa around 1.0-2.0 is often reasonable for small budgets.
    # Let's use a slightly adaptive approach or a well-tuned constant.
    # Given the small budget, aggressive exploration early might help.
    # However, UCB with a fixed kappa is robust.
    
    # Let's try kappa = 2.0 as a starting point for strong exploration.
    # Alternatively, we can use a beta-based exploration.
    
    # Let's stick with a simple UCB:
    # score = -mu + kappa * sigma
    
    kappa = 2.0
    score = -mu + kappa * sigma
    
    # Ensure finite scores
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
# EVOLVE-BLOCK-END
