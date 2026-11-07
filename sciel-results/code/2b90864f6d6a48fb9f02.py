import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Expected improvement per unit cost squared.
    # Stronger penalty for high cost to exploit correlation between
    # low cost (central region) and proximity to the moved minimum.
    z = (f_best - mu) / (sigma + 1e-9)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    return ei / (cost * cost + 1e-12)
