"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Hybrid acquisition: adaptive UCB with a small EI component for exploitation.
    Uses a time-based deadline guard to ensure we always return valid scores.
    """
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 300s limit
    
    n = len(mu)
    
    # Handle edge cases
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Compute standard normal CDF and PDF
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues
    z = np.clip(z, -10, 10)
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Upper Confidence Bound
    # Adaptive beta: start with moderate exploration, decay slightly
    # For 6D Hartmann, UCB with beta around 1.5-2.5 works well
    beta = 1.8
    
    ucb = mu + beta * sigma_safe
    
    # We want to MINIMIZE, so higher score = better (more promising)
    # For minimization, UCB for maximization would be: -mu + beta*sigma
    # But since we're minimizing and want to find low values, we should
    # use a lower confidence bound conceptually, but the acquisition 
    # function returns scores where HIGHER = evaluate next.
    
    # Actually, for minimization:
    # - EI is positive when mu < f_best (potential improvement)
    # - For UCB in minimization context, we want to explore where sigma is high
    #   and where mu is low. So we use: score = -mu + beta * sigma
    #   This gives high scores where mu is low (good) and sigma is high (uncertain)
    
    ucb_min = -mu + beta * sigma_safe
    
    # Normalize both to comparable scales
    # EI is already in objective units
    # ucb_min is also in objective units (negative mu + beta*sigma)
    
    # Blend: primarily UCB for exploration, some EI for exploitation
    # Weight: 70% UCB, 30% EI (tuned for 6D problem)
    w_ucb = 0.7
    w_ei = 0.3
    
    score = w_ucb * ucb_min + w_ei * ei
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > deadline - 5.0:
        # Return best effort immediately
        return score
    
    return score
# EVOLVE-BLOCK-END
