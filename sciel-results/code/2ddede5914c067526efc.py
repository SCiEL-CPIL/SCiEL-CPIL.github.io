import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    kappa = 2.0
    score = -mu + kappa * sigma
    return score
