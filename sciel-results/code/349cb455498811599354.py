import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # UCB (Upper Confidence Bound) for minimization
    # We want to maximize the score to pick the next point.
    # High score = Low mu (good) or High sigma (uncertain).
    # Score = -mu + beta * sigma
    
    beta = 2.5
    scores = -mu + beta * sigma
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
