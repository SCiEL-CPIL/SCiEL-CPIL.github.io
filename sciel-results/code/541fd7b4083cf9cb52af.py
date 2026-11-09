import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of hartmann 3d.
    Uses Lower Confidence Bound (LCB) adapted for maximization scoring.
    
    Args:
        mu: Array of mean predictions, shape (2000,).
        sigma: Array of standard deviation predictions, shape (2000,).
        f_best: The best (lowest) observed value so far.
        
    Returns:
        Array of scores, shape (2000,). Higher scores indicate better candidates.
    """
    # Beta controls exploration vs exploitation.
    # For minimization, LCB is mu - beta * sigma.
    # We want to minimize LCB, so we return -LCB = beta * sigma - mu.
    beta = 1.5
    
    scores = beta * sigma - mu
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=-1e10, posinf=1e10, neginf=-1e10)
    
    return scores
