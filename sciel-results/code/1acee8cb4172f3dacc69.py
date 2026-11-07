"""Acquisition function for Bayesian optimization of Ackley 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Track elapsed time across calls to enforce a deadline guard
_start_time = None

def acquisition(mu, sigma, f_best):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: if we've spent too much time, return a simple fallback
    elapsed = time.time() - _start_time
    if elapsed > 300:  # Safety margin before 330s limit
        return np.zeros_like(mu)
    
    # Ensure finite outputs
    sigma_safe = np.where(sigma > 1e-12, sigma, 1e-12)
    
    # Use a combination of Expected Improvement (EI) and Upper Confidence Bound (UCB)
    # EI for exploitation, UCB for exploration
    
    # Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # UCB: balance between exploitation and exploration
    # The beta parameter controls exploration; larger beta means more exploration
    # For Ackley 2D with 25 remaining evaluations after initial 5, use moderate exploration
    beta = 0.5
    ucb = mu + beta * sigma_safe
    
    # For minimization, we want to minimize, so we negate UCB for scoring
    # Higher score = more desirable to evaluate
    # UCB for minimization: we want low mu and high sigma (uncertainty)
    # So score = -ucb = -mu - beta * sigma_safe
    # But we also want to incorporate EI which is already positive when mu < f_best
    
    # Combine: primarily use EI (which is naturally positive for promising points)
    # Add a small exploration term based on sigma
    # The key insight: early on, we need more exploration; later, more exploitation
    
    # Use an adaptive blend: start with more exploration, shift to exploitation
    # Since we don't know the step number, use a fixed reasonable blend
    exploration_weight = 0.3
    
    scores = ei + exploration_weight * sigma_safe
    
    # Ensure finiteness and non-negativity
    scores = np.where(np.isfinite(scores), scores, 0.0)
    scores = np.maximum(scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
