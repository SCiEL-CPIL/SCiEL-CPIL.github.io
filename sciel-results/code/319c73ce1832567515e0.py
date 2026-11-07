"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 290.0  # 10s margin before 300s limit

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Cost-Adjusted UCB.
    
    Parameters:
    mu : np.ndarray, shape (2000,)
        Mean of the GP posterior.
    sigma : np.ndarray, shape (2000,)
        Standard deviation of the GP posterior.
    f_best : float
        Best observed value so far.
    cost : np.ndarray, shape (2000,)
        Cost of evaluating each candidate.
    
    Returns:
    scores : np.ndarray, shape (2000,)
        Acquisition scores (higher is better).
    """
    # Check deadline
    if time.time() > _DEADLINE:
        # Return a fallback score, e.g., based on low cost and high sigma
        # This is a safety net; ideally we should finish fast.
        return sigma / cost

    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle potential zero or negative sigma/cost
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Calculate z-score for EI (if we were using EI)
    # For UCB, we use: Score = (beta * sigma - mu) / cost
    # We want to minimize the function, so we look for low mu and high sigma.
    # The term (beta * sigma - mu) is high when sigma is high or mu is low.
    
    # Dynamic beta: start high for exploration, decrease for exploitation.
    # We don't know the exact iteration count, but we can infer it from the 
    # spread of sigma or just use a fixed robust beta.
    # A fixed beta of 2.0-3.0 is often good. Let's use 2.5.
    beta = 2.5
    
    # UCB score for minimization:
    # We want to maximize the potential for improvement.
    # Improvement = f_best - f(x).
    # Expected Improvement is complex.
    # UCB lower bound: L(x) = mu(x) - beta * sigma(x).
    # We want to minimize L(x).
    # So we maximize -L(x) = beta * sigma(x) - mu(x).
    
    ucb_score = beta * sigma - mu
    
    # Cost adjustment
    # Dividing by cost favors cheaper points.
    scores = ucb_score / cost
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    # Optional: Add a small exploration bonus if sigma is very high
    # This is already covered by beta * sigma.
    
    return scores
# EVOLVE-BLOCK-END
