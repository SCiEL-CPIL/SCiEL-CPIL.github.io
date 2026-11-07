"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state to track progress
_acq_call_count = 0
_start_time = None

def acquisition(mu, sigma, f_best, cost):
    global _acq_call_count
    global _start_time
    
    if _start_time is None:
        _start_time = time.time()
        
    _acq_call_count += 1
    
    # Estimate total cost spent
    # Initial 5 points + current call count
    # Average cost of a uniform point in [0,1]^2 scaled to [1,2] is 1.5
    # However, BO might pick cheaper points.
    # Let's assume average cost 1.4 to be slightly conservative (more progress)
    # or 1.5 for simplicity.
    n_evals = 5 + _acq_call_count
    est_total_cost = n_evals * 1.5
    max_cost = 30.0
    
    progress = est_total_cost / max_cost
    if progress > 1.0:
        progress = 1.0
    if progress < 0.0:
        progress = 0.0
        
    # Budget-scaled exploration coefficient
    # Decays linearly from beta_max to beta_min
    beta_max = 4.0
    beta_min = 0.5
    beta = beta_min + (beta_max - beta_min) * (1.0 - progress)
    
    # UCB for minimization: -mu + sqrt(beta) * sigma
    # We want to maximize this score
    sqrt_beta = np.sqrt(beta)
    ucb = -mu + sqrt_beta * sigma
    
    # Cost-aware normalization
    # Divide by cost. Cost is in [1, 2].
    # To avoid division by zero (though cost >= 1), we use cost directly.
    # The directive mentions "convex combination of cost and a constant".
    # Let's use denom = 0.5 * cost + 0.5 * 1.0 to soften the penalty slightly?
    # Or just cost?
    # Let's stick to cost for now, as it's the most direct "per unit cost" metric.
    # However, if cost is 1, score is UCB. If cost is 2, score is UCB/2.
    # This effectively doubles the "value" of cheap points.
    
    scores = ucb / cost
    
    # Ensure finite scores
    # If any NaN or Inf, replace with a very low value
    scores = np.where(np.isfinite(scores), scores, -1e9)
    
    return scores
# EVOLVE-BLOCK-END
