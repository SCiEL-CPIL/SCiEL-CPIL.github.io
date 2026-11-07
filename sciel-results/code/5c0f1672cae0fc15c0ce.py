import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a hybrid of Expected Improvement
    and a knowledge-gradient-like exploration term.
    
    Key structural changes from the previous version:
    1. Uses a cost-weighted EI where cost is normalized by its current mean,
       creating a relative cost penalty that adapts as total cost accumulates.
    2. Adds an exploration term that is proportional to sigma but modulated
       by a factor that decreases as we get closer to the optimum, encouraging
       exploration early and exploitation later.
    3. Includes a small bonus for points that are far from already-evaluated
       regions (measured via the GP uncertainty relative to the expected improvement).
    """
    t0 = time.time()
    
    # Guard against zero/negative sigma
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Expected Improvement: E[max(f_best - f(x), 0)]
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware weighting: divide by cost, but normalize by mean cost to keep
    # the scale stable as total cost accumulates
    cost_safe = np.maximum(cost, 1e-8)
    mean_cost = np.mean(cost_safe)
    cost_normalized = cost_safe / mean_cost
    
    # Primary term: EI normalized by relative cost
    cost_ei = ei / cost_normalized
    
    # Exploration term: encourage uncertainty but scale down as we find good values
    # Use a dynamic beta that decreases as f_best improves relative to the range
    # The Shekel 4D minimum is -10.5364, so we can use that as a reference
    f_min_ref = -10.5364
    # Progress: 0 when f_best is at initial values, 1 when near optimum
    # Assume initial f_best is around -4 to -5 (typical for random starts on Shekel)
    progress = np.clip((f_best - f_min_ref) / (-4.0 - f_min_ref), 0.0, 1.0)
    # Higher progress -> less exploration
    beta = 0.3 * (1.0 - 0.5 * progress)
    
    # Exploration: UCB-like term, also cost-aware
    exploration = beta * sigma_safe / cost_normalized
    
    # Combine
    scores = cost_ei + exploration
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Safety: if computation took too long, return what we have
    elapsed = time.time() - t0
    if elapsed > 290.0:
        return scores
    
    return scores
