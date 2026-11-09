import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
        # UCB with cost penalty
        beta = 2.0
        ucb = mu - beta * sigma
        # Avoid division by zero or very small cost
        cost = np.maximum(cost, 1e-6)
        return ucb / cost
