"""Cost-aware acquisition function for Bayesian optimization (cosine8 8d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Cost-Adjusted UCB (Upper Confidence Bound).
    This is a structural change from EI to a linear heuristic that balances
    exploitation (low mu) and exploration (high sigma) explicitly, normalized by cost.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        GP mean predictions.
    sigma : array-like, shape (2000,)
        GP standard deviation predictions.
    f_best : float
        Best observed function value.
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate.
    
    Returns:
    --------
    scores : array, shape (2000,)
        Acquisition scores (higher = more promising).
    """
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin for 300s limit
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma to avoid division by zero and ensure positivity
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Cost-Adjusted UCB for minimization
    # We want to minimize f(x).
    # Optimistic estimate of minimum: mu - beta * sigma
    # We want to maximize the potential improvement relative to cost.
    # Score = (f_best - (mu - beta * sigma)) / cost
    # This encourages points with low predicted mean (exploitation)
    # and high uncertainty (exploration), penalized by cost.
    
    # Beta controls exploration/exploitation trade-off.
    # A value of 2.0 is a common default for early-stage optimization.
    beta = 2.0
    
    # Calculate the numerator: potential improvement + exploration bonus
    # If mu is very high, this will be negative, which is fine (low score).
    numerator = f_best - mu + beta * sigma
    
    # Calculate scores
    scores = numerator / cost
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check deadline (unlikely to be hit for O(N) ops, but required by directive)
    if time.time() > deadline:
         # Return a valid fallback, e.g., uniform scores or zeros
         return np.zeros_like(scores)
        
    return scores
# EVOLVE-BLOCK-END
