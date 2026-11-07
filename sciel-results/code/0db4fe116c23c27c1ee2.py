import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 325.0  # 5s safety margin before 330s limit

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware UCB acquisition for Styblinski-Tang 2D BO.
    
    Parameters:
    - mu: array of shape (2000,), GP mean predictions
    - sigma: array of shape (2000,), GP std predictions
    - f_best: float, best observed function value
    - cost: array of shape (2000,), cost of evaluating each candidate
    
    Returns:
    - array of shape (2000,), scores (higher = evaluate next)
    """
    # Check deadline
    if time.time() > _DEADLINE:
        # Return deterministic fallback: prefer low cost, low mu
        return -mu / (cost + 1e-9)
    
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-9)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    mu = np.where(np.isfinite(mu), mu, f_best)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Adaptive exploration parameter beta
    # We don't have direct access to iteration count, but we can infer
    # from the relationship between f_best and the typical mu/sigma.
    # Early on, f_best will be high (far from optimum), sigma large.
    # Late on, f_best approaches -78.33, sigma small.
    # Use a heuristic based on how far f_best is from the known minimum.
    # Styblinski-Tang minimum is approximately -78.3323
    known_min = -78.3323
    
    # Normalize f_best to get a progress measure (0 = start, 1 = at optimum)
    # Initial uniform points on [-5,5]^2 give f values roughly in [-100, 100]
    # Let's use a robust estimate: if f_best is close to known_min, we're late
    progress = np.clip(1.0 - (f_best - known_min) / 200.0, 0.0, 1.0)
    
    # Beta schedule: start high (explore), decrease over time
    # At progress=0, beta ~ 3.0; at progress=1, beta ~ 0.5
    beta = 3.0 - 2.5 * progress
    
    # UCB score: mu + beta * sigma (we want to minimize, so we use -mu + beta*sigma)
    # Actually for minimization: score = -mu + beta * sigma
    # Higher score = more promising
    ucb = -mu + beta * sigma
    
    # Add a small exploitation floor: ensure that very promising (low mu) points
    # get a baseline score even if sigma is tiny, preventing pure exploration
    # from dominating. The floor is small so it doesn't override genuine UCB.
    # Scale by how close mu is to f_best: if mu << f_best, boost.
    exploitation_floor = 0.1 * np.maximum(0.0, (f_best - mu) / (np.abs(f_best) + 1.0))
    ucb = ucb + exploitation_floor
    
    # Cost-aware: divide by cost to penalize expensive evaluations
    # Adaptive cost exponent: stronger penalty early (explore cheap regions),
    # weaker penalty late (exploit even if slightly more expensive)
    cost_exp = 1.5 - 0.5 * progress  # 1.5 early, 1.0 late
    score = ucb / (cost ** cost_exp)
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
