import numpy as np
import time

def acquisition(mu, sigma, f_best):
    start_time = time.time()
    deadline = start_time + 10.0 # Safety margin
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-12)
    
    # UCB for minimization:
    # We want to minimize f(x).
    # The lower confidence bound is mu - beta * sigma.
    # We select the point with the lowest lower bound.
    # Therefore, the score to MAXIMIZE is the negative of the lower bound:
    # Score = -(mu - beta * sigma) = -mu + beta * sigma
    
    # Heuristic for beta:
    # In early stages (high uncertainty), we want larger beta.
    # We don't have the number of evaluations directly, but we can infer
    # from the spread of sigma or just use a fixed good value.
    # For Hartmann 3D, beta ~ 2.0 is a standard starting point.
    beta = 2.0
    
    scores = -mu + beta * sigma
    
    # Check time
    if time.time() > deadline:
        # If we are too slow (unlikely for simple vector ops), return something valid
        # Just ensure finite
        scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
        return scores
    
    # Ensure finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
