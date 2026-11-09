import numpy as np

def acquisition(mu, sigma, f_best, cost):
    # Use Upper Confidence Bound (UCB) for exploration
    # Score = (-mu + beta * sigma) / cost
    # Higher score means more promising (lower expected value, higher uncertainty, lower cost)
    beta = 3.0
    scores = (-mu + beta * sigma) / cost
    return scores
