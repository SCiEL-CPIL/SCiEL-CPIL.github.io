import numpy as np
import time

_start_time = time.time()
_DEADLINE = 330.0
_TIME_MARGIN = 10.0

def acquisition(mu, sigma, f_best, cost):
    # Time guard: if we're close to the deadline, return a safe fallback
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE - _TIME_MARGIN:
        # Return scores that favor low-cost, low-mu points as a fallback
        fallback = -mu / np.maximum(cost, 1e-9)
        return np.nan_to_num(fallback, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Numerical stability
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Cost-normalized Lower Confidence Bound (LCB)
    # LCB for minimization: mu - kappa * sigma (lower is better)
    # We want to maximize the score, so we use: -(mu - kappa * sigma) / cost
    # = (-mu + kappa * sigma) / cost
    
    # Exploration parameter tuned for small budget
    kappa = 1.5
    
    # LCB-based score: prefer points with low predicted mean and high uncertainty
    # per unit cost
    lcb_score = (-mu + kappa * sigma) / cost
    
    # Ensure finite output
    scores = np.nan_to_num(lcb_score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
