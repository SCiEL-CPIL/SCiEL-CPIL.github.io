"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = _start_time + 280  # 20s safety margin before 300s kill


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining Expected Improvement (EI)
    with a diversity/exploration term (UCB-like) to avoid premature convergence.
    
    The core idea: standard EI can be overly greedy, especially when the GP
    has low uncertainty in some regions. We add a small exploration bonus
    proportional to sigma (uncertainty) to encourage exploring uncertain regions,
    while still being cost-aware.
    
    Structure:
    1. Compute EI for minimization
    2. Compute exploration bonus from sigma
    3. Combine: score = (EI + beta * sigma) / cost
    4. Ensure finite outputs
    """
    # Deadline guard
    if time.time() > _DEADLINE:
        # Return a simple fallback: improvement over cost
        mu_safe = np.asarray(mu, dtype=np.float64)
        cost_safe = np.maximum(np.asarray(cost, dtype=np.float64), 1e-8)
        fallback = np.maximum(f_best - mu_safe, 0.0) / cost_safe
        return np.nan_to_num(fallback, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Ensure inputs are proper arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma and cost are positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)
    
    # Standardized improvement: how many std devs below f_best is mu?
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical overflow in norm.cdf/norm.pdf
    z = np.clip(z, -10.0, 10.0)
    
    # Expected Improvement for minimization:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Exploration bonus: encourages exploring regions with high uncertainty
    # Use a small coefficient to balance exploitation vs exploration
    # The coefficient 0.5 is chosen to be meaningful but not dominant
    beta = 0.5
    exploration_bonus = beta * sigma_safe
    
    # Combine EI and exploration bonus
    combined = ei + exploration_bonus
    
    # Cost-aware: divide by cost to prefer cheaper evaluations
    scores = combined / cost_safe
    
    # Ensure finite outputs
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
