import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Goldstein-Price 2D.
    Uses Upper Confidence Bound (UCB) to balance exploration and exploitation.
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Ensure sigma is strictly positive to avoid division by zero or numerical issues
    sigma = np.maximum(sigma, 1e-9)
    
    # Beta parameter for UCB.
    # A larger beta encourages exploration of high-uncertainty regions.
    # Given the small budget (30 evaluations), aggressive exploration is beneficial
    # to locate the global minimum.
    beta = 2.5
    
    # For minimization, we maximize the score: -mu + beta * sigma
    # This corresponds to the negative of the lower confidence bound.
    scores = -mu + beta * sigma
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
