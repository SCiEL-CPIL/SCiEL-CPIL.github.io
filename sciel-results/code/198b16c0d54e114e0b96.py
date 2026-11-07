import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Calculate z
    z = (f_best - mu) / (sigma + 1e-9)
    
    # Calculate EI
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Return EI (ignoring cost in the score to avoid biasing towards outer corners)
    return ei
