import numpy as np

def acquisition(mu, sigma, f_best):
    # UCB acquisition function for minimization
    # We want to maximize the score, so we negate the mean
    # and add the exploration term.
    kappa = 1.0
    score = -mu + kappa * sigma
    return score
