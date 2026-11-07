import numpy as np

def acquisition(mu, sigma, f_best):
    # Lower Confidence Bound (LCB) for minimization
    # We maximize the score, which corresponds to minimizing the lower bound.
    # Lower Bound = mu - beta * sigma
    # Score = -(mu - beta * sigma) = -mu + beta * sigma
    
    # Beta controls the exploration/exploitation trade-off.
    # A value around 2.0 to 4.0 is typical for 3D problems.
    beta = 3.0
    
    scores = -mu + beta * sigma
    
    # Ensure all scores are finite
    # mu and sigma should be finite from the GP, but just in case
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
