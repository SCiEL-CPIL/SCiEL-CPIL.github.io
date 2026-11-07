import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Upper Confidence Bound (UCB) acquisition function for minimization.
    We want to minimize f(x).
    The GP mean is mu, std dev is sigma.
    We estimate the potential of a point x as mu(x) - beta * sigma(x).
    To maximize the acquisition score (which indicates which point to pick next),
    we return the negative of this estimate: -mu(x) + beta * sigma(x).
    """
    beta = 1.5
    scores = -mu + beta * sigma
    return scores
