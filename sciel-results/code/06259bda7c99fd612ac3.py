# EVOLVE-BLOCK-START
import numpy as np

def acquisition(mu, sigma, f_best):
    # Upper Confidence Bound (UCB) for minimization
    # We want to maximize the score: -mu + beta * sigma
    # This balances exploitation (low mu) and exploration (high sigma)
    
    # Choose a beta parameter. 
    # beta = 1.0 corresponds to a 1-sigma confidence bound.
    # Higher beta encourages more exploration.
    beta = 1.0
    
    # Ensure sigma is non-negative to avoid issues with negative uncertainty
    sigma = np.maximum(sigma, 0.0)
    
    # Calculate UCB score
    # Minimizing f(x) is equivalent to maximizing -f(x)
    # So we use -mu as the expected value term
    scores = -mu + beta * sigma
    
    # Handle any potential numerical issues (e.g., inf or nan)
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
