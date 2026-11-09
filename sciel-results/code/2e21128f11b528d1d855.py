import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 300.0
_call_count = 0

def acquisition(mu, sigma, f_best, cost):
    global _call_count
    _call_count += 1

    # Deadline guard
    elapsed = time.time() - _START_TIME
    if elapsed > _TIME_LIMIT - 5.0:
        # Fallback: Greedy exploitation (maximize (0 - mu) / cost => -mu/cost)
        scores = -mu / np.maximum(cost, 1e-9)
        return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)

    # Dynamic exploration weight based on estimated remaining budget
    # Initial 5 points, uniform in [-4, 5]^4 -> scaled [0, 1]^4
    # Cost = 1 + mean(scaled_x). Mean cost ~ 1.5.
    # Total initial cost ~ 5 * 1.5 = 7.5.
    # Remaining budget initially ~ 22.5.
    
    # Estimate cost spent so far.
    # We assume the selected points have an average cost slightly lower than uniform
    # because the algorithm is cost-aware, but not too low. 1.3 is a reasonable guess.
    avg_step_cost = 1.3
    est_cost_so_far = 7.5 + avg_step_cost * _call_count
    
    rem_budget = 30.0 - est_cost_so_far
    # Normalize remaining budget to [0, 1]
    norm_rem = np.clip(rem_budget / 22.5, 0.0, 1.0)

    # UCB parameter beta
    # Start with high exploration, decay to low exploitation
    beta_max = 2.5
    beta_min = 0.5
    beta = beta_min + (beta_max - beta_min) * norm_rem

    # Cost-aware UCB for minimization
    # We want to minimize mu. Uncertainty sigma suggests potential for lower values.
    # Lower bound estimate: mu - beta * sigma.
    # We want to pick the candidate with the lowest lower bound.
    # Interface requires: Higher score = evaluate next.
    # So we return - (mu - beta * sigma) / cost = (beta * sigma - mu) / cost.
    
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    scores = (beta * sigma_safe - mu) / cost_safe
    
    # Ensure finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
