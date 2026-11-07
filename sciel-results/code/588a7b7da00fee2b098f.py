"""Rewritten acquisition for Bayesian optimization of Michalewicz 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE = 325.0

def acquisition(mu, sigma, f_best):
    """
    Acquisition function: Pure LCB with adaptive exploration.
    
    For minimization, LCB(x) = mu(x) + k * sigma(x).
    We maximize score = -LCB = -mu - k * sigma.
    
    We use an adaptive k that starts high (aggressive exploration) and
    decays as more evaluations are made (via f_best improving), allowing
    the optimizer to exploit promising regions later.
    """
    # Time guard
    if time.time() > _DEADLINE - 5.0:
        return -np.where(np.isfinite(mu), mu, 0.0)

    sigma_safe = np.maximum(sigma, 1e-8)

    # Adaptive exploration constant:
    # Start with high k for exploration, decay as f_best gets closer to optimum.
    # Michalewicz 2D minimum is ~-1.8013.
    # If f_best is still high (e.g., -1.0), we need more exploration.
    # If f_best is close to -1.8, we can exploit more.
    
    # Normalize: progress = (f_best - min_val) / (max_expected - min_val)
    # min_val = -1.8013, max_expected ~ 0 (random points can be near 0)
    # Actually, let's use a simpler heuristic based on how far f_best is from optimum.
    
    min_val = -1.8013
    # If f_best is very far from min, use high k. If close, use lower k.
    # Scale: when f_best = -1.0, progress is large. When f_best = -1.7, progress is small.
    # Let's define a decay based on (f_best - min_val).
    # Early on, f_best might be ~ -1.0 to -1.5. Later, ~ -1.7 to -1.8.
    
    # Use a smooth transition:
    # k = k_min + (k_max - k_min) * exp(-(f_best - min_val) * beta)
    # But f_best - min_val is always positive (since f_best >= min_val).
    # If f_best is close to min_val, (f_best - min_val) is small, exp is ~1, k ~ k_min + (k_max-k_min) = k_max.
    # Wait, that's backwards. We want high k when f_best is FAR from min_val.
    # So: k = k_min + (k_max - k_min) * exp(-(min_val - f_best) * beta) ? No.
    # Let's use: distance = f_best - min_val.
    # Large distance -> high k. Small distance -> low k.
    # k = k_min + (k_max - k_min) * (distance / distance_scale)
    # Or simply: k = k_max * exp(-distance / decay_scale)
    
    distance = f_best - min_val
    # If distance is large, k should be large.
    # Let's use k = 1.0 + 2.0 * exp(-distance * 2.0)
    # If distance = 0.5 (f_best = -1.3), k = 1.0 + 2.0 * exp(-1.0) = 1.0 + 0.73 = 1.73
    # If distance = 0.1 (f_best = -1.7), k = 1.0 + 2.0 * exp(-0.2) = 1.0 + 1.63 = 2.63
    # If distance = 0.01 (f_best = -1.79), k = 1.0 + 2.0 * exp(-0.02) = 1.0 + 1.96 = 2.96
    
    # Actually, we want the OPPOSITE. We want to EXPLOIT when we are close to the optimum.
    # So k should DECREASE as distance decreases.
    # k = k_min + (k_max - k_min) * exp(-distance / decay_scale)
    # If distance is large, exp is small, k ~ k_min.
    # If distance is small, exp is large, k ~ k_max.
    # This is still backwards.
    
    # Let's try: k = k_max * exp(-distance / decay_scale)
    # If distance is large, k is small. If distance is small, k is large.
    # This is also backwards.
    
    # Correct logic:
    # We want to explore MORE when we are FAR from the optimum (high uncertainty about where it is).
    # We want to exploit MORE when we are CLOSE to the optimum (we know where it is).
    # So:
    # Far from optimum (large distance) -> High k (explore).
    # Close to optimum (small distance) -> Low k (exploit).
    
    # k = k_min + (k_max - k_min) * (distance / distance_scale)
    # If distance is large, k is large. If distance is small, k is small.
    # Let's use distance_scale = 0.5 (typical range of f_best - min_val).
    # k_min = 0.5, k_max = 3.0.
    
    distance_scale = 0.5
    k_min = 0.5
    k_max = 3.0
    
    # Clip distance to avoid huge k if f_best is very bad.
    distance_clipped = np.clip(distance, 0, 1.0)
    k = k_min + (k_max - k_min) * (distance_clipped / distance_scale)
    
    # Ensure k is within bounds
    k = np.clip(k, k_min, k_max)
    
    # LCB score: higher is better
    score = -(mu + k * sigma_safe)
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
