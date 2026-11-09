import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 290.0

def _check_deadline():
    return time.time() > _DEADLINE

def acquisition(mu, sigma, f_best, cost):
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    cost = np.asarray(cost)
    
    if _check_deadline():
        # Fallback: simple EI / cost
        sigma_safe = np.where(sigma > 1e-9, sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        from scipy.stats import norm
        ei = np.maximum((f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z), 0.0)
        return ei / (cost + 1e-9)

    sigma_safe = np.where(sigma > 1e-9, sigma, 1e-9)
    
    # Adaptive Beta for UCB
    # Scale beta based on current best value.
    # Assume typical Ackley values are in [0, 30].
    # If f_best is high, explore more (high beta).
    # If f_best is low, exploit more (low beta).
    # Normalize f_best to a 0-1 range for scaling.
    # Let's use a soft scaling.
    # beta = base_beta * (f_best / reference_value)
    # Reference value for Ackley 2D max is ~30-40. Let's use 10 as a threshold.
    # If f_best > 10, beta is high.
    # If f_best < 1, beta is low.
    
    # A simple adaptive beta:
    # beta = 5.0 * np.exp(-f_best / 5.0) + 0.5
    # This gives beta ~5.5 at start (f_best large), and beta ~0.5 at end.
    
    beta = 5.0 * np.exp(-f_best / 5.0) + 0.5
    
    # UCB for minimization: mu - beta * sigma
    ucb = mu - beta * sigma_safe
    
    # Cost-aware score
    scores = ucb / (cost + 1e-9)
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e9, neginf=-1e9)
    
    return scores
