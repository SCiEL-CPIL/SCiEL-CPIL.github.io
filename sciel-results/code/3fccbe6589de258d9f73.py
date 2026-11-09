import numpy as np

def acquisition(mu, sigma, f_best):
    # UCB for minimization: maximize -mu + beta * sigma
    # We return scores where higher is better.
    beta = 2.5
    scores = -mu + beta * sigma
    return scores
