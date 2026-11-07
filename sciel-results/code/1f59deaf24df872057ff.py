"""Direct cost-aware acquisition for Bayesian optimization of rosenbrock 2d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition using a Thompson Sampling-inspired stochastic score.
    
    Strategy:
    1. Draw a single random sample from the GP posterior: s = mu - sigma * z,
       where z ~ N(0, 1). This represents a plausible function value at each candidate.
    2. Score each candidate as the ratio of the improvement over f_best to its cost:
       score = max(0, f_best - s) / cost
    3. This effectively treats the objective as a stochastic reward-to-go ratio,
       naturally balancing exploration (via the random sample) and exploitation
       (via the improvement term) while accounting for evaluation cost.
    
    We guard against the 330s wall-clock limit with an explicit deadline.
    """
    start_time = time.time()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: handle NaN/Inf
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=1e-6, posinf=1e6, neginf=1e-6)
    cost = np.nan_to_num(cost, nan=1.0, posinf=2.0, neginf=1.0)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-8)
    
    # Draw a single random sample from the GP posterior
    # s = mu - sigma * z, where z ~ N(0, 1)
    z = np.random.standard_normal(size=mu.shape)
    s = mu - sigma * z
    
    # Compute the improvement over f_best (positive if s < f_best)
    improvement = f_best - s
    
    # Ensure improvement is non-negative
    improvement = np.maximum(improvement, 0.0)
    
    # Cost-aware score: improvement per unit cost
    safe_cost = np.maximum(cost, 1e-8)
    score = improvement / safe_cost
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > 300:  # Additional safety check
        # Fallback: deterministic Expected Improvement per cost
        z_ei = (f_best - mu) / sigma
        ei = (f_best - mu) * norm.cdf(z_ei) + sigma * norm.pdf(z_ei)
        ei = np.maximum(ei, 0.0)
        score = ei / safe_cost
    
    # Ensure finite
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
# EVOLVE-BLOCK-END
