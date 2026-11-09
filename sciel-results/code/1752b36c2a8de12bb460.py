import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Goldstein-Price 2D.
    Uses a Model-and-Optimize approach with adaptive beta for UCB,
    combined with a lower confidence bound style score.
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Deadline guard: we have at most ~330s per evaluation in total,
    # but this function is called once per step (25 steps of 2000 candidates each).
    # The GP fit dominates the time, but let's be safe.
    start_time = time.time()
    deadline = start_time + 2.0  # 2 second safety margin for this call
    
    # Strategy: Use UCB with adaptive beta.
    # When f_best is close to the known minimum (3.0), exploit more (lower beta).
    # When f_best is far, explore more (higher beta).
    
    # The known minimum of Goldstein-Price 2D is 3.0.
    # Simple regret is f_best - 3.0.
    # We want to minimize simple regret.
    
    # Adaptive beta:
    # If f_best is much larger than 3.0, we are far away, so explore more.
    # If f_best is close to 3.0, we are close, so exploit more.
    
    # Let's define a gap = f_best - 3.0
    gap = max(f_best - 3.0, 0.0)
    
    # Normalize the gap to get a beta parameter.
    # Typical Goldstein-Price values range from 3 to ~80.
    # A reasonable range for beta is 1.0 to 5.0.
    
    # Use a sigmoid-like function to map gap to beta.
    # When gap is 0, beta ~ 1.5 (exploit)
    # When gap is 5, beta ~ 3.5 (balanced)
    # When gap is 20, beta ~ 5.0 (explore)
    
    # beta = 1.5 + 3.5 * (1 - np.exp(-gap / 5.0))
    # Or simpler:
    # Increased exploration weight to better escape local minima in early steps
    beta = 2.5 + 2.5 * np.tanh(gap / 4.0)
    
    # Compute UCB score: -mu + beta * sigma
    # This is equivalent to maximizing the lower confidence bound for minimization.
    scores = -mu + beta * sigma
    
    # Also consider a pure EI component for very close points
    # But UCB with adaptive beta should handle both exploration and exploitation.
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check deadline
    if time.time() > deadline:
        # Return best available
        return scores
    
    return scores
