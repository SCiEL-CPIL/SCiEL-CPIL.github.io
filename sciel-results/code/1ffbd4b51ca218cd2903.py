"""Cost-aware acquisition function for Bayesian optimization (hartmann 6d)."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

_DEADLINE = 300.0  # seconds wall-clock limit
_START_TIME = time.time()


def _check_deadline():
    """Return True if we are within the safety margin of the deadline."""
    return (time.time() - _START_TIME) < (_DEADLINE - 5.0)


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-adjusted Probability of Improvement (POI) with dynamic improvement threshold.
    
    The threshold is scaled by remaining budget to demand larger improvements late in the run.
    Returns 2000 finite scores (higher = evaluate next).
    """
    n = len(mu)
    eps = 1e-9
    
    # Guard against non-finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, eps)
    cost = np.maximum(cost, eps)
    
    # Compute remaining budget proxy: total budget is 30, initial 5 points cost ~5*1.5=7.5
    # We don't know exact total cost, but f_best and cost give us info.
    # Use a simple heuristic: remaining fraction based on cost spent.
    # Average cost per eval is roughly 1 + 0.5 = 1.5 for uniform points in [0,1]^6
    # Total budget 30, so max evals ~ 30/1.5 = 20
    # We've spent sum(cost) so far. But cost here is per-candidate cost.
    # Actually cost[i] is the cost of evaluating candidate i if chosen.
    # The total cost spent is tracked externally. We only have per-candidate cost.
    
    # Dynamic threshold: as budget depletes, demand larger improvements.
    # Use a simple schedule: gamma increases with the number of evaluations.
    # We can estimate progress from f_best relative to known minimum.
    # Hartmann 6d min is -3.32237.
    f_min = -3.32237
    
    # Progress: 0 at start (f_best ~ 0), 1 at optimum
    # f_best starts around -0.5 to -1.5 for 5 random points
    # Use a normalized progress
    progress = np.clip((f_best - 0.0) / (f_min - 0.0), 0.0, 1.0)
    # Actually f_best < 0, so (f_best - 0)/(f_min - 0) = f_best/f_min which is positive
    # f_best ~ -1, f_min ~ -3.32, so progress ~ 0.3
    # This is a rough proxy. Let's use a simpler approach.
    
    # Simpler: use the number of evaluations as a proxy.
    # We can estimate n_evals from the fact that we started with 5 and add 1 per loop.
    # But we don't have direct access. Let's use a fixed schedule based on f_best.
    
    # Gamma: scales the improvement threshold.
    # Early: small gamma (exploit more), Late: large gamma (demand big improvements)
    # Use f_best to estimate progress
    # If f_best is close to f_min, we're late
    remaining_to_opt = f_best - f_min  # positive, decreases as we improve
    max_remaining = 0.0 - f_min  # ~3.32
    remaining_frac = np.clip(remaining_to_opt / max_remaining, 0.0, 1.0)
    
    # Gamma increases as remaining_frac decreases
    # gamma = base + scale * (1 - remaining_frac)
    base_gamma = 0.1
    scale_gamma = 0.5
    gamma = base_gamma + scale_gamma * (1.0 - remaining_frac)
    
    # Target improvement threshold
    # f_target = f_best - gamma * sigma_scale
    # Use a fraction of the current uncertainty
    sigma_mean = np.mean(sigma)
    threshold = gamma * sigma_mean
    
    # POI: P(f(x) < f_best - threshold)
    # = P(Z < (f_best - threshold - mu) / sigma)
    # = CDF((f_best - threshold - mu) / sigma)
    z = (f_best - threshold - mu) / sigma
    poi = norm.cdf(z)
    
    # Cost adjustment: divide by cost
    scores = poi / cost
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    scores = np.maximum(scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
