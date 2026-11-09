import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: ensure we return well before 300s
    # This is a single acquisition call, so it should be fast.
    # But we add a check just in case of pathological inputs.
    t0 = time.time()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure shapes are correct
    n = 2000
    if mu.shape != (n,):
        mu = mu.reshape(-1)
        if mu.shape[0] != n:
            # Fallback: return zeros if shape is wrong
            return np.zeros(n, dtype=np.float64)
    if sigma.shape != (n,):
        sigma = sigma.reshape(-1)
        if sigma.shape[0] != n:
            return np.zeros(n, dtype=np.float64)
    if cost.shape != (n,):
        cost = cost.reshape(-1)
        if cost.shape[0] != n:
            return np.zeros(n, dtype=np.float64)
    
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    cost_safe = np.maximum(cost, eps)
    
    # Structural Change: Switch from EI to UCB (Upper Confidence Bound) for minimization.
    # EI is often too greedy for multi-modal functions like Levy 2D.
    # UCB encourages exploration more strongly.
    # Acquisition for minimization (to be maximized):
    # A(x) = (f_best - mu) + kappa * sigma
    # This is equivalent to minimizing (mu - kappa * sigma).
    
    kappa = 2.0
    ucb_score = (f_best - mu) + kappa * sigma_safe
    
    # Cost-awareness: Divide by cost.
    # Lower cost is better.
    scores = ucb_score / cost_safe
    
    # Check for non-finite values
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    # If all scores are -inf, return zeros to avoid invalid results
    if not np.any(np.isfinite(scores)):
        return np.zeros(n, dtype=np.float64)
    
    # Check time
    if time.time() - t0 > 280:
        # Return best effort
        return scores
    
    return scores
