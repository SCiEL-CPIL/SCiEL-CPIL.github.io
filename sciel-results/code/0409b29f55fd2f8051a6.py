import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # UCB with beta = 2.5
    # For minimization, we want low mu and high sigma
    # Score = (-mu + beta * sigma) / cost
    beta = 2.5
    scores = (-mu + beta * sigma) / cost
    return scores
