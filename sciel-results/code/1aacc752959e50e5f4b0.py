import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # UCB for minimization: we want to minimize f(x).
    # The "optimistic" lower bound is mu - kappa * sigma.
    # We want to maximize the potential for improvement.
    # Score = (kappa * sigma - mu) / cost
    
    # Alternatively, use EI but with a different cost normalization?
    # Let's try UCB first.
    
    kappa = 3.0
    ucb = kappa * sigma - mu
    # Handle numerical issues
    scores = ucb / cost
    
    # Ensure finite
    scores[~np.isfinite(scores)] = -np.inf
    return scores
