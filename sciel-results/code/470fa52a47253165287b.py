import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Cost-aware Upper Confidence Bound
    # We want to minimize f.
    # Optimistic value: mu - beta * sigma (lower is better)
    # Score: (f_best - (mu - beta * sigma)) / cost
    # This rewards points that are likely to be better than f_best, 
    # with a bonus for uncertainty.
    
    beta = 2.0
    # Add small epsilon to cost just in case, though cost >= 1
    safe_cost = cost + 1e-9
    score = (f_best - mu + beta * sigma) / safe_cost
    
    # Ensure finite scores
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
