import numpy as np

def acquisition(mu, sigma, f_best):
    # Upper Confidence Bound (UCB) for minimization
    # We want to maximize the score.
    # Score = -mu + beta * sigma
    # High score -> Low mu (exploitation) or High sigma (exploration)
    beta = 2.0
    score = -mu + beta * sigma
    return score
