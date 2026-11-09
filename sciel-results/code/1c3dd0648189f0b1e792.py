"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Combined acquisition: Thompson Sampling-inspired stochastic score
    blended with a deterministic LCB term for stability.
    
    The Thompson Sampling component samples from the posterior N(mu, sigma^2)
    and scores candidates by their negative sampled value (lower objective is better).
    To avoid pure randomness dominating, we blend with a deterministic LCB term
    that provides consistent exploitation pressure.
    """
    start_time = time.time()
    deadline = start_time + 280  # Safety margin before 330s kill

    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Ensure sigma is positive and finite
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-8)
    
    # Handle non-finite mu
    mu = np.where(np.isfinite(mu), mu, f_best + 1.0)
    
    # Thompson Sampling: sample once from posterior
    # Use a fixed random seed per call? No, we want stochasticity across calls.
    # But for a single call, we draw one sample.
    # To make it more robust, we can average a few samples or use the median.
    # However, a single sample is the canonical TS.
    # Let's try a hybrid: score = -sampled_value + exploration_bonus
    
    # Draw Thompson sample
    rng = np.random.default_rng()
    z = rng.standard_normal(size=mu.shape)
    sampled_values = mu + sigma * z
    
    # The score for Thompson sampling is the negative of the sampled value
    # (since lower objective is better, we want higher score for lower values)
    ts_scores = -sampled_values
    
    # Deterministic LCB component for stability
    # LCB = mu - beta * sigma
    # Score = -LCB = -mu + beta * sigma
    # We use a modest beta to not overwhelm the TS component
    beta = 0.5
    lcb_scores = -mu + beta * sigma
    
    # Blend the two components
    # We want to leverage the exploration of TS but stabilize with LCB
    # A simple weighted average in score space
    # Normalize both to have similar scales? 
    # TS scores can be very spread out. LCB scores are more bounded.
    # Let's use a weighted sum where TS provides the primary signal
    # but LCB adds a consistent bias toward low mu / high sigma
    
    # To prevent TS from being too noisy, we can clip extreme samples
    # or use a robust statistic. But let's try a direct blend first.
    # Weight 0.7 for TS, 0.3 for LCB
    score = 0.7 * ts_scores + 0.3 * lcb_scores
    
    # Check deadline
    if time.time() > deadline:
        # If we're running out of time, just return the LCB scores which are deterministic and fast
        score = lcb_scores
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, -1e10)
    
    return score
# EVOLVE-BLOCK-END
