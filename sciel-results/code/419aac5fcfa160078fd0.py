import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # UCB (Upper Confidence Bound) acquisition
    # Balances exploration (sigma) and exploitation (mu)
    # beta controls the exploration-exploitation tradeoff
    beta = 2.0
    score = mu + beta * sigma
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    return score
