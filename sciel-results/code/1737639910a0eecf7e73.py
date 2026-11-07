import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Use UCB for acquisition
    # Cost is in [1, 2]
    # Beta controls exploration. 
    # Given the GP smoothness mismatch, high exploration is needed.
    beta = 4.0
    # UCB for minimization: -mu + beta * sigma
    ucb = -mu + beta * sigma
    # Cost-aware: divide by cost
    # Ensure cost is not zero (it's >= 1)
    scores = ucb / cost
    return scores
