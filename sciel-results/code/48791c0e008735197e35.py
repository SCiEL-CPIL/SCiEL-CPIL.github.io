import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Use UCB for minimization
    # Score = (beta * sigma - mu) / cost
    # We want to maximize this score.
    # Low mu (good) -> high score.
    # High sigma (uncertain) -> high score.
    # Low cost (cheap) -> high score.
    
    beta = 2.0
    scores = (beta * sigma - mu) / cost
    
    # Ensure finite
    return scores
