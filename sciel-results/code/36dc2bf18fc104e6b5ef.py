import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Cost-aware Lower Confidence Bound for minimization.
    # We want to minimize f(x).
    # The lower bound is mu - beta * sigma.
    # We want to select points with low lower bounds.
    # Since the acquisition score should be "higher is better",
    # we return the negative of the lower bound, scaled by cost.
    # Score = -(mu - beta * sigma) / cost = (beta * sigma - mu) / cost
    
    # Choose beta. A value around 2-5 is common for small budgets.
    # Let's try beta = 3.0 to encourage exploration.
    beta = 3.0
    
    # Ensure sigma is positive for numerical stability, though GP sigma should be > 0.
    # The task says sigma is a standard deviation.
    
    scores = (beta * sigma - mu) / cost
    
    # Handle any potential non-finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
