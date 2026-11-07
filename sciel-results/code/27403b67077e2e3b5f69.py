"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global deadline guard
_start_time = time.time()
_DEADLINE = _start_time + 320  # 320s out of 330s budget

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Thompson Sampling with a deterministic
    UCB fallback blended for stability.
    
    Structural change: Thompson Sampling draws a single sample from the GP
    posterior (mu - sigma * z) for the entire candidate batch. The acquisition
    score is the negative of the sampled value divided by cost. This naturally
    balances exploration and exploitation without heuristic tuning of beta.
    We blend with a small UCB component to ensure deterministic structure
    is preserved in the selection, which can help with stability across
    different GP states.
    """
    # Check deadline
    if time.time() > _DEADLINE:
        # Fallback: simple cost-aware mean
        cost_safe = np.maximum(cost, 1e-9)
        return -mu / cost_safe
    
    # Ensure inputs are clean
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.maximum(sigma, 1e-9)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Thompson Sampling: draw a single standard normal sample z
    # This is a scalar that applies to all candidates, creating a
    # correlated sample from the GP posterior.
    z = np.random.standard_normal()
    
    # Sampled posterior: y = mu - sigma * z
    # For minimization, lower y is better.
    # We want to maximize score, so score = -y / cost = (-mu + sigma * z) / cost
    ts_score = (-mu + sigma * z) / cost
    
    # Blend with a deterministic UCB (beta=2.0) for stability
    # This prevents the random sample from occasionally making poor choices
    # when the GP is uncertain in many places simultaneously.
    beta = 2.0
    ucb_score = (-mu + beta * sigma) / cost
    
    # Weighted combination: primarily Thompson Sampling, with some UCB
    # The weight on UCB can be small, e.g., 0.2, to provide structure
    w_ts = 0.8
    w_ucb = 0.2
    score = w_ts * ts_score + w_ucb * ucb_score
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, -1e10)
    
    return score
# EVOLVE-BLOCK-END
