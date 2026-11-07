import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Cost-aware UCB
    # We want to minimize f(x).
    # We use the Lower Confidence Bound (LCB) of the function value as a proxy for "how good it could be".
    # LCB = mu - beta * sigma.
    # We want to pick the point with the LOWEST LCB (potentially the best minimum).
    # So we want to MAXIMIZE the score: -LCB = -mu + beta * sigma.
    # We divide by cost to account for the budget.
    
    beta = 2.5
    
    # Ensure cost is not zero (it's >= 1)
    score = (beta * sigma - mu) / cost
    
    return score
