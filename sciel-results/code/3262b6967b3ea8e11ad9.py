import numpy as np

def acquisition(mu, sigma, f_best):
    # Upper Confidence Bound (UCB) for minimization.
    # We transform the minimization problem into a maximization of -f(x).
    # The posterior mean of -f(x) is -mu(x) and std is sigma(x).
    # UCB for maximization: score = mean + beta * std
    # score = -mu(x) + beta * sigma(x)
    beta = 2.0
    scores = -mu + beta * sigma
    return scores
