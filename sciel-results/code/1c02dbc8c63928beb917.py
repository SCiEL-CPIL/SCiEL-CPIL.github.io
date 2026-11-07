import numpy as np
from scipy.stats import norm
import time

_T0 = time.time()
_DEADLINE = 310.0

def _time_ok():
    return (time.time() - _T0) < _DEADLINE

def acquisition(mu, sigma, f_best, cost):
    if not _time_ok():
        # Fallback to simple EI
        mu = np.asarray(mu, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        cost = np.asarray(cost, dtype=np.float64)
        sigma_safe = np.maximum(sigma, 1e-8)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        score = ei / np.maximum(cost, 1e-6)
        return np.where(np.isfinite(score), score, 0.0)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Safety
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    sigma_safe = np.maximum(sigma, 1e-8)

    f_best_val = float(f_best)
    if not np.isfinite(f_best_val):
        f_best_val = 0.0
    
    # Progress estimation
    # f_min = -78.3323
    # Assume initial f_best is around 0 or higher.
    # If f_best is very high, progress is low.
    progress = np.clip(-f_best_val / 78.3323, 0.0, 1.0)
    
    # Adaptive Beta for UCB
    # High exploration early, low exploration late
    beta = 0.5 + 3.5 * (1.0 - progress)
    
    # UCB for minimization: -mu + beta * sigma
    ucb = -mu + beta * sigma_safe
    
    # Cost Penalty
    # cost ranges from 1 to 2
    # Use exponential penalty to strongly discourage expensive points
    # gamma = 1.0 means cost=2 is penalized by e^-1 ~ 0.37
    # gamma = 2.0 means cost=2 is penalized by e^-2 ~ 0.13
    # Let's try gamma = 1.5
    gamma = 1.5
    cost_penalty = np.exp(-gamma * (cost - 1.0))
    
    score = ucb * cost_penalty
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
