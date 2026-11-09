import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Upper Confidence Bound (UCB) acquisition function for minimization.
    
    Parameters:
    mu: array-like, shape (n_candidates,)
        Mean of the Gaussian Process posterior.
    sigma: array-like, shape (n_candidates,)
        Standard deviation of the Gaussian Process posterior.
    f_best: float
        The best (minimum) value observed so far.
        
    Returns:
    scores: array-like, shape (n_candidates,)
        UCB scores. Higher is better.
    """
    # Ensure sigma is positive to avoid issues
    sigma = np.maximum(sigma, 1e-9)
    
    # For minimization, we want low mu and high sigma (exploration)
    # UCB for minimization: score = -mu + beta * sigma
    # We can also normalize by considering how far below f_best we expect to go
    
    # A common approach for minimization is:
    # UCB = -(mu - beta * sigma) = -mu + beta * sigma
    
    # Let's use a beta that balances exploration and exploitation
    # The optimal beta depends on the problem, but a good starting point is around 1-2
    
    # We can also use a dynamic beta based on the number of evaluations,
    # but since we don't have that information, we'll use a fixed value
    
    beta = 1.5
    
    # Compute UCB scores for minimization
    # We want to maximize this score
    scores = -mu + beta * sigma
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
