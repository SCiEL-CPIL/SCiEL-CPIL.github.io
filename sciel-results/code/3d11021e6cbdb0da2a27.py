import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Use a target slightly higher than f_best to encourage finding significantly better points
    # This avoids getting stuck in local minima by ignoring marginal improvements
    f_target = f_best + 0.5
    z = (f_target - mu) / (sigma + 1e-9)
    return (f_target - mu) * norm.cdf(z) + sigma * norm.pdf(z)
