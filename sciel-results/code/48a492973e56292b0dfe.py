import numpy as np

def acquisition(mu, sigma, f_best):
    # UCB for minimization
    # We want to minimize the function value.
    # The lower confidence bound is mu - kappa * sigma.
    # We select the point with the lowest lower bound.
    # To return a score where higher is better, we return - (mu - kappa * sigma).
    kappa = 1.5
    scores = -mu + kappa * sigma
    return scores
