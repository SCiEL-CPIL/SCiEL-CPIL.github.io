import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # UCB for minimization: we want to minimize f(x).
    # Optimistic value is mu - beta * sigma.
    # We want to maximize the "potential improvement" over f_best.
    # Improvement = f_best - (mu - beta * sigma) = f_best - mu + beta * sigma.
    # However, standard UCB score for minimization is often just:
    # Score = - (mu - beta * sigma) / cost  OR  (beta * sigma - mu) / cost
    
    # Let's use: Score = (beta * sigma - mu) / cost
    # This is high when mu is low (good) and sigma is high (uncertain).
    # It is normalized by cost.
    
    beta = 3.0
    scores = (beta * sigma - mu) / cost
    
    # Ensure finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
