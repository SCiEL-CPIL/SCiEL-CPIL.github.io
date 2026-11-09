"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Cost-aware Upper Confidence Bound.
    # UCB: mu + beta * sigma, normalized by cost.
    # beta controls exploration; higher beta explores more.
    beta = 1.8
    # Ensure sigma is non-negative and handle numerical issues
    sigma_safe = np.maximum(sigma, 1e-8)
    ucb = mu + beta * sigma_safe
    # Normalize by cost to prefer cheaper evaluations
    # cost is already provided as (2000,)
    cost_safe = np.maximum(cost, 1e-8)
    score = ucb / cost_safe
    # Return finite scores; higher = evaluate next
    return score
# EVOLVE-BLOCK-END
