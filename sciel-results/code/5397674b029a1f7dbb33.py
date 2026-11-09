"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Hybrid acquisition: combines Expected Improvement (EI) with an exploration
    bonus based on uncertainty, and a penalty for points that are already very
    close to f_best (to encourage exploration of unexplored regions).
    
    We use a modified EI that adds a small exploration term proportional to
    sigma, and subtracts a term that penalizes candidates whose mu is very close
    to or below f_best (since we've likely already explored there).
    
    Key structural change: Use a "knowledge gradient" inspired approach with
    explicit exploration/exploitation balance, plus a deadline guard.
    """
    start_time = time.time()
    deadline = start_time + 295  # Safety margin before 300s limit
    
    # Clip sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standard EI
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Exploration bonus: reward high uncertainty
    # Scale exploration to be meaningful relative to EI values
    # Use a decaying exploration factor
    exploration_weight = 0.3
    
    # Exploration term: proportional to sigma, but bounded
    sigma_max = np.max(sigma_safe)
    if sigma_max > 1e-9:
        exploration_bonus = exploration_weight * sigma_safe / sigma_max * (f_best - np.min(mu))
    else:
        exploration_bonus = np.zeros_like(sigma_safe)
    
    # Penalty for candidates that are already very good (mu close to f_best)
    # This encourages the algorithm to explore rather than exploit known good points
    # Use a soft penalty: penalize when mu is within a small fraction of f_best
    # This is like a "novelty" bonus
    
    # Distance from f_best in standardized units
    dist_from_best = (f_best - mu) / sigma_safe
    
    # Penalize points that are already very close to the best (exploited)
    # and reward points that are uncertain
    exploit_penalty = 0.1 * np.exp(-dist_from_best ** 2 / 2.0) * (f_best - np.min(mu))
    
    # Combine: EI + exploration - exploitation penalty
    scores = ei + exploration_bonus - exploit_penalty
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Deadline check - if we're running long, return what we have
    if time.time() > deadline:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
