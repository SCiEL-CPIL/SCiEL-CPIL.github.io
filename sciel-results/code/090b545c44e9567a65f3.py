import numpy as np
from scipy.stats import norm
import time

_start_time = None
_DEADLINE = 320.0  # Safety margin before 330s kill

def acquisition(mu, sigma, f_best, cost):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: if we're running too long, return a simple fallback
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        # Fallback: return normalized sigma as exploration score
        sigma = np.asarray(sigma, dtype=float).ravel()
        cost = np.asarray(cost, dtype=float).ravel()
        cost = np.clip(cost, 1e-12, None)
        return np.where(np.isfinite(sigma / cost), sigma / cost, 0.0)
    
    # Ensure inputs are 1D arrays
    mu = np.asarray(mu, dtype=float).ravel()
    sigma = np.asarray(sigma, dtype=float).ravel()
    cost = np.asarray(cost, dtype=float).ravel()
    
    # Safety margins
    sigma = np.clip(sigma, 1e-12, None)
    cost = np.clip(cost, 1e-12, None)
    
    # Probability of Improvement (minimization): P(f(x) < f_best)
    # For normal distribution: P(mu + sigma*Z < f_best) = Phi((f_best - mu)/sigma)
    delta = f_best - mu
    z = delta / sigma
    
    # Clip z for numerical stability
    z = np.clip(z, -10.0, 10.0)
    
    # PoI = Phi(z)
    poi = norm.cdf(z)
    
    # Cost-aware score: maximize PoI / cost + small uncertainty bonus
    # This prioritizes candidates with high probability of success per unit cost
    # The uncertainty bonus (sigma/cost) helps with exploration in multi-modal landscapes
    # Weight of 0.15 balances exploitation (PoI/cost) and exploration (sigma/cost)
    score = poi / cost + 0.15 * (sigma / cost)
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
