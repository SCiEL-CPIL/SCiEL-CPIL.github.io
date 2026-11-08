import numpy as np

def acquisition(mu, sigma, f_best, cost):
    # Lower Confidence Bound (for minimization)
    # We want to minimize mu - beta * sigma
    # So we maximize beta * sigma - mu
    # Divide by cost to account for cost
    beta = 1.0
    scores = (beta * sigma - mu) / cost
    return scores
