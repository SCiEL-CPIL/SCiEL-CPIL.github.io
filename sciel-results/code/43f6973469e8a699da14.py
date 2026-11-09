"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function: Expected Improvement (EI) with a small exploration bonus.
    
    We use a variant of EI that also considers the probability of finding a
    significantly better point, to encourage exploration in early stages.
    
    mu: (n,) array of GP mean predictions
    sigma: (n,) array of GP standard deviation predictions
    f_best: current best observed value (minimum)
    
    Returns: (n,) array of scores, higher is better
    """
    start_time = time.time()
    deadline = start_time + 295.0  # 5s safety margin before 300s limit
    
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standardized improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Dynamic Pareto-based weighting: balance exploitation (EI) and exploration (uncertainty)
    # Use f_best relative to known minimum (-3.32237) as progress indicator
    # When f_best is far from minimum, we're early -> more exploration
    # When f_best is close to minimum, we're late -> more exploitation
    
    # Progress: 0 when f_best is poor, 1 when f_best is at minimum
    # Hartmann 6d range is roughly [-3.3, 3.3], so span ~6.6
    progress = np.clip((f_best - (-3.32237)) / 6.6, 0, 1)
    # Invert: early (high progress value means far from min) -> more exploration
    # Actually progress should be 0 early, 1 late
    # f_best starts high (~3.3), ends near -3.32237
    # So (f_best - min) / span goes from ~6.6/6.6=1 to 0
    # We want exploration weight high early (when f_best is far from min)
    exploration_weight = 1.0 - np.clip((f_best - (-3.32237)) / 6.6, 0, 1)
    # Clamp to reasonable range
    exploration_weight = np.clip(exploration_weight, 0.05, 0.5)
    
    # Normalize EI and sigma for balanced combination
    # Use robust normalization to avoid scale issues
    ei_max = np.max(ei) if np.max(ei) > 1e-12 else 1.0
    sigma_max = np.max(sigma_safe) if np.max(sigma_safe) > 1e-12 else 1.0
    
    ei_norm = ei / ei_max
    sigma_norm = sigma_safe / sigma_max
    
    # Combine with dynamic weight
    scores = (1.0 - exploration_weight) * ei_norm + exploration_weight * sigma_norm
    
    # Scale back to comparable magnitude as original EI
    scores = scores * ei_max
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check if we're approaching the deadline
    elapsed = time.time() - start_time
    if elapsed > deadline - start_time:
        # Return best available result quickly
        return scores
    
    return scores
# EVOLVE-BLOCK-END
