import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Calculate EI
    z = (f_best - mu) / (sigma + 1e-9)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Calculate UCB term (potential improvement)
    # Using a beta that encourages exploration
    beta = 2.0
    ucb = f_best - mu + beta * sigma
    
    # Hybrid: We want to ensure we don't ignore pure exploitation if we are close to the global min
    # But we also want to explore.
    # A common trick is to use the maximum of the two, or a weighted sum.
    # Since EI is bounded by ~sigma, and UCB can be larger, let's use a weighted sum.
    # If we are far from the optimum, EI is large. UCB is large.
    # If we are at a local min, EI is 0. UCB is beta*sigma.
    # So UCB drives exploration at local minima.
    
    # Let's try a simple weighted sum, tuned to favor UCB slightly more than EI
    # because the seed (EI) is the baseline and UCB improved it.
    score = (0.5 * ei + 0.5 * ucb) / cost
    
    return score
