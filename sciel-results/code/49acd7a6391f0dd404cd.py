import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 320.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition for Styblinski-Tang 2D BO.
    Uses an ensemble of acquisition functions (EI, UCB) to robustly select the next point.
    """
    t0 = time.time()
    
    # Safety: ensure inputs are clean
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Replace non-finite values
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma) & (sigma > 1e-12), sigma, 1e-9)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Deadline guard check early
    if time.time() > _DEADLINE - 5.0:
        # Fallback to simple UCB
        beta_fallback = 2.5
        score = (-mu + beta_fallback * sigma) / cost
        return np.where(np.isfinite(score), score, 0.0)

    # Progress estimate
    known_min = -78.3323
    progress = np.clip(1.0 - (f_best - known_min) / 250.0, 0.0, 1.0)
    
    # 1. Expected Improvement (EI)
    z = (f_best - mu) / (sigma + 1e-12)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # 2. UCB with adaptive beta
    # Early: high exploration (high beta), Late: low exploration
    beta = 1.0 + 3.0 * (1.0 - progress)
    ucb = -mu + beta * sigma
    
    # 3. UCB with fixed high beta for robust exploration
    ucb_high = -mu + 4.0 * sigma
    
    # Combine by taking the maximum. This allows the algorithm to switch
    # between exploitation (EI) and exploration (UCB) based on which signal is stronger
    # at each point, effectively creating a piecewise acquisition surface.
    combined_score = np.maximum(ei, np.maximum(ucb, ucb_high))
    
    # Cost adjustment
    # Penalize expensive points.
    score = combined_score / cost
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
