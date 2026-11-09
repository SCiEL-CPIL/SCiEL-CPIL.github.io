import numpy as np

def acquisition(mu, sigma, f_best):
    kappa = 2.0
    score = -mu - kappa * sigma
    return score
