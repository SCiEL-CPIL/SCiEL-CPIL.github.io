import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Upper Confidence Bound (UCB) for minimization.
    We maximize the score, so we use the Lower Confidence Bound (LCB) of the function value.
    LCB = mu - beta * sigma
    """
    beta = 2.0
    return mu - beta * sigma
