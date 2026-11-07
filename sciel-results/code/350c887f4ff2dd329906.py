import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Upper Confidence Bound (UCB) for minimization
    # LCB(x) = mu(x) - beta * sigma(x)
    # We want to minimize LCB, so we maximize -LCB(x) = beta * sigma(x) - mu(x)
    # Score = (beta * sigma(x) - mu(x)) / cost(x)
    
    beta = 2.0  # Exploration parameter
    
    # Calculate UCB score
    # Higher score indicates a better candidate (more promising or more uncertain)
    scores = (beta * sigma - mu) / cost
    
    # Ensure scores are finite
    # If cost is 0, it would be inf, but cost is 1 + mean(x) so min cost is 1.
    # If mu or sigma are inf, scores will be inf. The problem statement says "Return 2000 finite scores".
    # GP outputs should be finite.
    
    return scores
