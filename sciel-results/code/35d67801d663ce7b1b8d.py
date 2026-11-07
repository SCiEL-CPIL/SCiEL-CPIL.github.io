import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # UCB for minimization: minimize mu - beta * sigma
    # So maximize -mu + beta * sigma
    beta = 2.0
    scores = (beta * sigma - mu) / cost
    return scores
