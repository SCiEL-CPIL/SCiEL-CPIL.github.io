import numpy as np

def acquisition(mu, sigma, f_best):
    # Upper Confidence Bound (UCB) for minimization.
    # Encourages exploration of high-uncertainty regions to prevent
    # collapse into local minima, acting as a diversity-aware strategy
    # by favoring areas not yet well-explored by the GP.
    
    # Exploration parameter
    beta = 1.0
    
    # Score = -mu + beta * sigma
    # We want to minimize the function, so lower mu is better.
    # Higher sigma is better for exploration.
    # We return scores where higher is better for the evaluator to pick.
    scores = -mu + beta * sigma
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
