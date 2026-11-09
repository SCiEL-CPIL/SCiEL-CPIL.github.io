import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Cost-aware Confidence Bound (CB)
    # Objective: Minimize f(x)
    # Lower Bound: L(x) = mu(x) - beta * sigma(x)
    # Acquisition: Maximize (-L(x)) / cost(x)
    
    # Beta controls exploration vs exploitation.
    # Higher beta -> more exploration.
    # Given the tight budget and 2D space, we need significant exploration.
    beta = 2.5
    
    # Ensure numerical stability
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Calculate the score
    # -mu + beta * sigma
    numerator = -mu + beta * sigma
    scores = numerator / cost
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
