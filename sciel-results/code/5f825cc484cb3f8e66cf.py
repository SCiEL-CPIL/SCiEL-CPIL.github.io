"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: 330s wall-clock limit, return early if close
    t0 = time.time()
    deadline = t0 + 325  # 5s safety margin
    
    # 1. Safety checks and clipping
    # Ensure sigma is strictly positive for log/exp operations
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # 2. Budget-aware exploration scaling
    # Total budget is 30.
    # remaining_budget = 30 - cost
    # As cost approaches 30, remaining_budget -> 0, so beta -> 0 (exploitation).
    # At start (cost ~5), remaining_budget ~ 25, beta is large (exploration).
    # We use sqrt to make the decay gradual initially and sharp near the end.
    remaining_budget = np.maximum(30.0 - cost, 0.0)
    
    # Scale beta to be meaningful in objective units.
    # Hartmann 6d range is roughly [-3.3, -3.0] for the minimum, but values can be higher.
    # A beta of ~1-2 is typical for LCB in standardized space, but here mu/sigma are in objective units.
    # Let's scale it such that at full budget, beta is around 2.0, and it drops to 0.
    # beta = 2.0 * sqrt(remaining_budget / 25.0)
    # This ensures beta is 2.0 at cost=5 and 0 at cost=30.
    beta = 2.0 * np.sqrt(remaining_budget / 25.0)
    
    # 3. Compute LCB
    # LCB = mu - beta * sigma
    # We want to minimize, so higher LCB is better (less negative).
    lcb = mu - beta * sigma_safe
    
    # 4. Cost-awareness
    # The cell finding suggests cost penalty exponent 0.5 is better than 1.
    # score = lcb / cost^0.5
    # This penalizes high cost but less severely than linear division.
    cost_safe = np.maximum(cost, 1e-8)
    cost_penalty = np.power(cost_safe, 0.5)
    
    score = lcb / cost_penalty
    
    # 5. Ensure finite scores
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
# EVOLVE-BLOCK-END
