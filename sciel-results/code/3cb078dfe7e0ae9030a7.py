import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 320.0

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard
    if time.time() - _START_TIME > _TIME_LIMIT:
        cost_safe = np.maximum(cost, 1e-9)
        return -mu / cost_safe

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)

    # Aggressive exploration early on
    # f_best starts high (~100-130). 
    # Previous: kappa = 1 + f/50. f=100 -> 3.
    # New: kappa = 1 + f/20. f=100 -> 6.
    # This encourages larger jumps when we are far from the optimum.
    kappa = 1.0 + f_best / 20.0
    kappa = np.clip(kappa, 1.0, 10.0)

    std = kappa * sigma
    y = np.random.normal(mu, std)

    scores = -y / cost
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
