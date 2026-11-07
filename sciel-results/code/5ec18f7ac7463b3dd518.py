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
    if elapsed > 310:  # Safety margin before 330s limit
        return np.zeros_like(mu)
    
    # Ensure finite outputs
    sigma_safe = np.where(sigma > 1e-12, sigma, 1e-12)
    
    # Use UCB with a tuned beta for exploration
    # For minimization, we want low mu and high sigma
    # UCB for minimization: score = -(mu + beta * sigma)
    # But we want higher scores to be better, so: score = -mu - beta * sigma
    # However, we can also use a form that directly gives higher scores for promising points
    
    # Expected Improvement for minimization:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z) where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # UCB-based score: we want to explore high uncertainty regions
    # For minimization, a good score is -mu + beta * sigma (we want low mu, high sigma)
    # Let's use a combination: primarily EI with an exploration bonus
    
    # Adaptive beta: start with more exploration, decrease over time
    # Since we don't know the exact step, use a moderate fixed beta
    beta = 1.0
    
    # UCB score for minimization: -mu + beta * sigma
    # This gives high scores to points with low mean and high uncertainty
    ucb_score = -mu + beta * sigma_safe
    
    # Combine EI and UCB
    # EI is naturally positive when mu < f_best
    # UCB score can be positive or negative depending on mu and sigma
    
    # Normalize or scale appropriately
    # Let's use a weighted combination
    ei_weight = 0.7
    ucb_weight = 0.3
    
    # Scale UCB score to be comparable to EI
    # EI is in objective units, UCB score is also in objective units (roughly)
    scores = ei_weight * ei + ucb_weight * ucb_score
    
    # Ensure finiteness and non-negativity
    scores = np.where(np.isfinite(scores), scores, 0.0)
    scores = np.maximum(scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
