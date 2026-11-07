"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def acquisition(mu, sigma, f_best):
    # Use Upper Confidence Bound (UCB) for minimization.
    # We want to maximize the score to select the next point.
    # For minimization, the UCB value is mu - beta * sigma.
    # To get a score where higher is better, we return -UCB = -mu + beta * sigma.
    
    # Exploration parameter
    beta = 2.0
    
    # Calculate UCB score
    scores = beta * sigma - mu
    
    return scores
# EVOLVE-BLOCK-END
