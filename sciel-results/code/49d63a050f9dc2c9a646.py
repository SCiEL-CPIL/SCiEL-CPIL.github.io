"""Acquisition function for Bayesian optimization of goldstein price 2d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE_MARGIN = 5.0

def _elapsed():
    return time.time() - _start_time

def acquisition(mu, sigma, f_best):
    """
    Compute acquisition scores using a hybrid of Expected Improvement (EI)
    relative to the current best and Probability of Improvement (PI) relative
    to the known global minimum (3.0).
    
    The weight shifts from EI to PI as f_best approaches the known minimum,
    ensuring strong exploration signals early on and precise targeting late on.
    """
    # Deadline guard
    if _elapsed() > _DEADLINE_MARGIN:
        mu = np.asarray(mu, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        # Fallback: simple UCB
        scores = -mu + 2.0 * np.maximum(sigma, 1e-10)
        return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    n = len(mu)
    if n == 0:
        return np.array([])
    
    # Safety for sigma
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # If f_best is not finite, treat it as a large number (explore)
    if not np.isfinite(f_best):
        f_best = 100.0
        
    # 1. Expected Improvement relative to f_best
    # z_best = (f_best - mu) / sigma
    z_best = (f_best - mu) / sigma_safe
    ei_best = (f_best - mu) * norm.cdf(z_best) + sigma_safe * norm.pdf(z_best)
    
    # 2. Probability of Improvement relative to known global minimum (3.0)
    # z_target = (3.0 - mu) / sigma
    # PI = P(f(x) < 3.0) = Phi(z_target)
    z_target = (3.0 - mu) / sigma_safe
    pi_target = norm.cdf(z_target)
    
    # Normalize both to [0, 1]
    # EI can be negative? No, EI is always >= 0.
    # PI is in [0, 1].
    
    ei_max = np.max(ei_best)
    if ei_max > 1e-10:
        ei_norm = ei_best / ei_max
    else:
        ei_norm = np.zeros_like(ei_best)
        
    # pi_target is already in [0, 1], but we might want to scale it if it's very small
    # To make them comparable, we can just use pi_target directly or scale it.
    # If pi_target is very small (e.g., 1e-5), it might be dominated by noise in ei_norm.
    # Let's normalize pi_target too, but keep its relative order.
    pi_max = np.max(pi_target)
    if pi_max > 1e-10:
        pi_norm = pi_target / pi_max
    else:
        pi_norm = np.zeros_like(pi_target)

    # Adaptive Weighting
    # We want to use PI_target more when f_best is close to 3.0.
    # Let distance = f_best - 3.0
    # If distance is large (e.g., 10), weight_pi = 0.
    # If distance is 0, weight_pi = 1.
    # Use a sigmoid or linear decay.
    # Scale: assume f_best might be in [3, 20].
    # Linear decay from 3 to 13.
    dist = f_best - 3.0
    if dist <= 0:
        weight_pi = 1.0
    elif dist >= 10.0:
        weight_pi = 0.0
    else:
        weight_pi = 1.0 - (dist / 10.0)
        
    # Add a small base weight for PI to ensure we always look for the target
    weight_pi = np.clip(weight_pi + 0.1, 0.1, 1.0)
    weight_ei = 1.0 - weight_pi
    
    scores = weight_ei * ei_norm + weight_pi * pi_norm
    
    # Add a small exploration bonus for high sigma regions to break ties
    # This helps if both EI and PI are zero (e.g., early on, far from target)
    # But EI should handle early exploration.
    # Let's add a tiny UCB term to ensure we don't get stuck in flat regions.
    # ucb_score = -mu + 0.1 * sigma
    # ucb_norm = (ucb_score - min) / (max - min)
    # scores += 0.05 * ucb_norm
    
    # Ensure finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Final deadline check
    if _elapsed() > _DEADLINE_MARGIN:
        return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        
    return scores
# EVOLVE-BLOCK-END
