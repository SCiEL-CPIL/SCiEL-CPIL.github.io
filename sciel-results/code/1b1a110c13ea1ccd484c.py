import numpy as np

def acquisition(mu, sigma, f_best):
    # Lower Confidence Bound for minimization.
    # We want to maximize this score to select the next point.
    # Score = -mu + beta * sigma
    # This is equivalent to minimizing (mu - beta * sigma).
    beta = 1.5
    scores = -mu + beta * sigma
    return scores
