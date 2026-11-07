import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # UCB for minimization
    # Score = Exploration - Exploitation
    # Exploration: sqrt(beta) * sigma
    # Exploitation: mu (we want low mu, so -mu in score)
    
    # Choose beta
    # A common heuristic is beta = 2 * log(1 + t)
    # Without t, let's use a fixed value.
    # 2.0 is a reasonable start.
    beta = 2.0
    
    scores = np.sqrt(beta) * sigma - mu
    
    # Ensure finite scores
    # If sigma or mu are inf/nan, handle them?
    # The problem states "Return 2000 finite scores".
    # We should replace non-finite values with very negative numbers (so they are not picked)
    # or very positive?
    # Higher score = evaluate next.
    # If a point is invalid (nan), we should not pick it.
    # So set score to -np.inf or a very small number.
    scores[~np.isfinite(scores)] = -1e10
    
    return scores
