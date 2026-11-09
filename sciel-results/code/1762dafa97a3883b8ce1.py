import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Probability of Improvement (PI)
    # For minimization, we want P(f(x) < f_best)
    # z = (f_best - mu) / sigma
    # P(f(x) < f_best) = Phi(z)
    z = (f_best - mu) / (sigma + 1e-9)
    return norm.cdf(z)
