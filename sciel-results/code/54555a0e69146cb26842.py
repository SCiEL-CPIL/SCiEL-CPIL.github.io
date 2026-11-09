"""Rewritten program using UCB (Upper Confidence Bound) for better exploration in multimodal landscapes."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware UCB acquisition function for Bayesian optimization.
    
    Uses Upper Confidence Bound (UCB) to balance exploitation (low mu) 
    and exploration (high sigma), penalized by evaluation cost.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        GP mean predictions at candidate points.
    sigma : array-like, shape (2000,)
        GP standard deviation predictions at candidate points.
    f_best : float
        Best observed function value so far.
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = more promising).
    """
    start_time = time.time()
    deadline = start_time + 280  # Safety margin before 300s limit
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = len(mu)
    if n == 0:
        return np.array([])
    
    # Ensure sigma is non-negative and safe for calculations
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # UCB Parameter (beta)
    # A value of 2.0 provides a good balance between exploration and exploitation
    # for problems of this dimensionality.
    beta = 2.0
    
    # Calculate UCB score for minimization.
    # We want to minimize mu - beta * sigma.
    # To convert to a maximization problem (higher score is better):
    # Score = (beta * sigma - mu)
    ucb_term = beta * sigma_safe - mu
    
    # Cost-awareness: Divide by cost to favor cheaper evaluations.
    # Cost is in the range [1, 2] based on the problem description (1 + mean(x)).
    cost_safe = np.maximum(cost, 1e-8)
    scores = ucb_term / cost_safe
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > deadline:
        # Return best effort
        pass
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
