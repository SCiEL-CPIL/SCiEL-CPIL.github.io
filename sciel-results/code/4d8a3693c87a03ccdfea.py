import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    z = (f_best - mu) / (sigma + 1e-9)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    # Reduce the impact of cost to allow exploration in more expensive (but potentially better) regions
    return ei / (cost ** 0.5)
