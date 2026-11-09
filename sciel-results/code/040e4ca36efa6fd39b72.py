"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard: track start time for the first call in this process
_start_time = [None]
_call_count = [0]

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using adaptive PI with exploration decay.
    """
    if _start_time[0] is None:
        _start_time[0] = time.time()
    elapsed = time.time() - _start_time[0]
    if elapsed > 325:
        # Safe fallback: prefer low mu, low cost
        mu_safe = np.where(np.isfinite(mu), mu, 0.0)
        cost_safe = np.where(np.isfinite(cost), cost, 1.0)
        return -mu_safe / (cost_safe + 1e-9)
    
    _call_count[0] += 1
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Sanitize
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    sigma = np.maximum(sigma, 1e-9)
    cost = np.where(np.isfinite(cost), cost, 1.5)
    mu = np.where(np.isfinite(mu), mu, f_best)
    
    # Estimate progress: total budget 30, initial 5 points ~ cost 7.5
    # Each subsequent eval costs ~1.5, so ~15 more evals possible
    # Use call_count to estimate fraction of budget used
    n_calls = _call_count[0]
    # Approximate: after n calls, we've used ~7.5 + (n-1)*1.5 cost if n>1
    # But simpler: fraction = n / 15 (rough estimate of remaining evals)
    frac = min(1.0, n_calls / 15.0)
    
    # Probability of Improvement: P(f(x) < f_best)
    z = (f_best - mu) / sigma
    z = np.clip(z, -50, 50)
    pi = norm.cdf(z)
    
    # Cost-normalized PI
    pi_cost = pi / (cost + 1e-9)
    
    # Adaptive exploration: start with high exploration, decay to exploitation
    # Exploration term: sigma / cost, scaled by (1 - frac) to decay
    exploration_weight = 0.5 * (1.0 - frac)
    exploration = exploration_weight * sigma / (cost + 1e-9)
    
    # Combine: PI (exploitation) + exploration
    scores = pi_cost + exploration
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
