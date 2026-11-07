"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition function for Bayesian optimization of Ackley 2D.
    Draws a single random sample from the posterior at each candidate point and
    scores based on the negative of that sample (lower is better, so we negate
    to make higher scores mean 'evaluate next').
    
    The key insight: Thompson Sampling naturally balances exploration and exploitation
    through stochastic sampling. When sigma is large (uncertain regions), the sample
    can be either very high or very low, creating natural exploration. When sigma
    is small (well-explored regions), the sample is close to mu, favoring exploitation
    of known good regions.
    
    To adapt to the optimization progress, we modulate the exploration by scaling
    sigma based on how good f_best is. Early on (f_best large), we want more exploration;
    late (f_best near 0), we want more exploitation.
    """
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-9)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    
    # Adaptive exploration scale:
    # When f_best is large (early), we want more exploration -> scale sigma up
    # When f_best is small (late), we want exploitation -> scale sigma down
    # f_best for Ackley typically starts around 2-4 and goes to 0
    
    # Use a smooth schedule: 
    # exploration_scale = 1.0 when f_best > 1.0 (explore freely)
    # exploration_scale decreases as f_best approaches 0
    # This prevents the GP from being too conservative in late stages
    
    f_best_clamped = max(f_best, 0.0)
    
    # Schedule: 
    # f_best >= 1.0: scale = 1.0 (full exploration)
    # f_best = 0.0: scale = 0.3 (reduced exploration, more exploitation)
    # Linear interpolation between 0 and 1
    if f_best_clamped >= 1.0:
        exploration_scale = 1.0
    else:
        exploration_scale = 0.3 + 0.7 * f_best_clamped
    
    # Draw a single standard normal sample for all candidates
    # This is the key Thompson Sampling trick: one random draw per acquisition call
    z = np.random.standard_normal(len(mu))
    
    # Posterior sample: theta = mu + scaled_sigma * z
    theta = mu + exploration_scale * sigma * z
    
    # We want to MINIMIZE, so lower theta is better.
    # But the acquisition should return HIGHER scores for candidates to evaluate next.
    # So we negate: score = -theta
    scores = -theta
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
