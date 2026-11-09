import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Computes the Lower Confidence Bound (LCB) acquisition function for minimization.
    
    Parameters:
    mu : array-like, shape (N,)
        Posterior mean at N candidate points.
    sigma : array-like, shape (N,)
        Posterior standard deviation at N candidate points.
    f_best : float
        Current best observed function value.
        
    Returns:
    scores : ndarray, shape (N,)
        LCB scores for the candidate points. Higher scores indicate more promising points.
    """
    # LCB for minimization: we want to minimize mu - beta * sigma
    # So the acquisition score should be negative of that, i.e., -mu + beta * sigma
    # Or equivalently, we can return mu - beta * sigma and the caller picks the minimum,
    # but the task says "higher meaning evaluate next", so we want higher scores for better points.
    # Thus, score = -mu + beta * sigma
    
    # Choose beta to balance exploration and exploitation
    # A common choice is beta = 1.0 or slightly larger
    beta = 1.5
    
    # Compute LCB score: higher is better
    scores = -mu + beta * sigma
    
    return scores
