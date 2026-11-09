"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Cost-aware Upper Confidence Bound (UCB) for minimization.
    # We maximize the score to select the next point.
    # The optimistic estimate of the function value (lower is better) is mu - beta * sigma.
    # We want to minimize this, so we maximize its negative: beta * sigma - mu.
    # We divide by the cost to account for the budget constraint.
    
    beta = 2.5
    score = (beta * sigma - mu) / cost
    return score
# EVOLVE-BLOCK-END
