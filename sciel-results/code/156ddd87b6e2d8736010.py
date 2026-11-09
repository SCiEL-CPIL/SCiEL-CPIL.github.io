import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Upper Confidence Bound (UCB) for minimization.
    
    Args:
        mu: numpy array of shape (N,) posterior mean.
        sigma: numpy array of shape (N,) posterior standard deviation.
        f_best: float, the best observed function value so far.
        
    Returns:
        numpy array of shape (N,) scores. Higher scores indicate better candidates.
    """
    # Deadline guard: start timer
    start_time = time.time()
    
    # UCB parameter (beta)
    # Tuned to balance exploration and exploitation for Styblinski-Tang
    # Higher beta encourages more exploration of high uncertainty regions
    beta = 3.0
    
    # For minimization, we want to select points with low mu and high sigma.
    # Score = -mu + beta * sigma
    # This is the negative of the Lower Confidence Bound (LCB) of the function,
    # or equivalently, the Upper Confidence Bound of the negative function.
    scores = -mu + beta * sigma
    
    # Safety margin check
    # If we are taking too long (unlikely for vectorized ops, but required), return
    if time.time() - start_time > 290:
        return scores

    # Ensure all scores are finite to avoid invalid results
    # Replace NaN with 0, Inf with large finite number
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
