import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # UCB for minimization: we want to maximize (beta * sigma - mu)
    # This balances exploration (sigma) and exploitation (low mu)
    beta = 1.5
    ucb_scores = beta * sigma - mu
    # Normalize by cost to get "value per unit cost"
    # Ensure cost is not zero (it's >= 1)
    scores = ucb_scores / cost
    return scores
