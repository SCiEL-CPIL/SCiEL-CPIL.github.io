import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Small exploration term to avoid premature convergence
    xi = 0.01
    z = (f_best - mu - xi) / (sigma + 1e-9)
    # Expected Improvement with xi
    return (f_best - mu - xi) * norm.cdf(z) + sigma * norm.pdf(z)
