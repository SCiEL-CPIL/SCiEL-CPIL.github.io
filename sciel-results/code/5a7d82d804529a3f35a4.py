import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    z = (f_best - mu) / (sigma + 1e-9)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    return ei
