import numpy as np
import time
from scipy.stats import norm

# Global deadline guard
_START_TIME = time.time()
_LIMIT = 285.0  # Safety margin below 300s

def acquisition(mu, sigma, f_best, cost):
    # Deadline check
    if time.time() - _START_TIME > _LIMIT:
        return np.zeros_like(mu)

    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Sanitize inputs
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)

    # Cost-Aware Upper Confidence Bound (UCB) for minimization
    # We want to minimize f(x).
    # UCB for minimization: mu - beta * sigma
    # We want to maximize the score, so we use: beta * sigma - mu
    # This rewards high uncertainty (exploration) and low mean (exploitation).
    
    # Adaptive beta:
    # A fixed beta is simple, but an adaptive one can be better.
    # Let's start with a fixed beta that encourages exploration.
    # For Griewank, which is multimodal, exploration is crucial.
    # Standard GP-UCB uses beta = sqrt(2 * log(t)).
    # Since we don't have t, we estimate it or use a heuristic.
    # Let's use beta = 1.0 as a baseline for structural change from EI.
    # If sigma is large, this term dominates. If mu is low, this term dominates.
    
    beta = 1.0
    
    # Calculate UCB score (higher is better for selection)
    ucb_score = beta * sigma - mu
    
    # Cost-awareness: Divide by cost
    # This normalizes the expected gain by the cost of evaluation.
    # To prevent division by zero or very small costs causing explosion,
    # we ensure cost is at least 1.0 (which it is by definition: 1 + mean(x)).
    
    score = ucb_score / cost
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
