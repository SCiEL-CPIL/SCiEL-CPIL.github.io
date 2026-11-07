"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_deadline = _start_time + 285.0
_call_count = [0]  # Mutable state to track number of acquisition calls

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of ackley 2d.
    Uses Adaptive UCB: dynamically balances exploration and exploitation
    based on the number of evaluations performed so far.
    
    Score_i = -mu_i + beta * sigma_i
    where beta decreases from a high initial value (exploration) to a
    low final value (exploitation) as more evaluations are made.
    
    Higher scores correspond to better candidates to evaluate next.
    """
    # Deadline guard: if we are running out of time, fall back to greedy (minimize mean)
    if time.time() > _deadline:
        return -mu

    n = len(mu)
    
    # Ensure sigma is non-negative and finite
    sigma = np.asarray(sigma, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    
    # Sanitize inputs
    sigma = np.maximum(sigma, 1e-10)  # Avoid zero variance
    sigma = np.nan_to_num(sigma, nan=1e-10, posinf=1e-10, neginf=1e-10)
    mu = np.nan_to_num(mu, nan=f_best, posinf=f_best, neginf=f_best)
    
    # Track step count (each call to acquisition corresponds to one evaluation step)
    step = _call_count[0]
    _call_count[0] += 1
    
    # Total expected evaluations: 5 initial + 25 optimization steps = 30
    # But we only see calls during the 25 optimization steps (after initial 5)
    # Actually, the loop runs until 30 evaluations including first 5, so
    # there are 25 calls to acquisition.
    total_steps = 25
    
    # Adaptive beta: starts high (exploration) and decays to low (exploitation)
    # Use a smooth decay: beta = beta_max * (1 - step/total_steps)^power
    # beta_max controls initial exploration intensity
    # power controls how quickly we shift to exploitation
    
    beta_max = 2.0  # Initial exploration scale
    power = 1.5     # Decay rate
    
    # Linear interpolation from beta_max to beta_min
    beta_min = 0.3  # Residual exploration even at the end
    progress = min(step / max(total_steps - 1, 1), 1.0)
    beta = beta_max + (beta_min - beta_max) * (progress ** power)
    
    # UCB score: we want to minimize, so higher score = better
    # score_i = -mu_i + beta * sigma_i
    # This favors candidates with low mean (exploitation) and high uncertainty (exploration)
    scores = -mu + beta * sigma
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=-f_best, posinf=1e10, neginf=-1e10)
    
    return scores
# EVOLVE-BLOCK-END
