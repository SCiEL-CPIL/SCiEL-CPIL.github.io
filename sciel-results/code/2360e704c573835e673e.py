import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Hybrid acquisition function: Expected Improvement + Exploration
    # EI provides exploitation signal
    # sigma term provides exploration signal
    z = (f_best - mu) / (sigma + 1e-9)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    # Combine with exploration bonus
    # Coefficient 1.0 balances exploration similar to UCB xi=1.0
    score = ei + 1.0 * sigma
    return score
