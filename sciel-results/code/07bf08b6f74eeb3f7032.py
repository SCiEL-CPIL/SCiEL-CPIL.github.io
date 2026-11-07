"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = None

def acquisition(mu, sigma, f_best):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard
    elapsed = time.time() - _start_time
    if elapsed > 310:
        # Fallback: simple UCB
        sigma_safe = np.maximum(sigma, 1e-9)
        return -mu + 1.0 * sigma_safe
    
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute standard Expected Improvement for minimization
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Entropy-weighted EI (H-EI): multiplicative interaction
    # The entropy weight amplifies EI in high-uncertainty regions
    # and suppresses it in well-explored (low-uncertainty) regions.
    # This is structurally different from additive shifts or UCB blends.
    
    # Normalize sigma to [0, 1] for the entropy weight
    sigma_max = np.max(sigma_safe)
    if sigma_max > 1e-9:
        sigma_norm = sigma_safe / sigma_max
    else:
        sigma_norm = np.zeros_like(sigma_safe)
    
    # Entropy-like weight: use a smooth function that emphasizes mid-to-high uncertainty
    # H(sigma) = sigma_norm * (1 - sigma_norm * 0.3) 
    # This peaks at moderate-high uncertainty and tapers at extremes
    # Alternatively, use the normalized sigma directly as a multiplicative factor
    # with a floor to avoid zeroing out low-uncertainty regions entirely
    
    # Use a power-law entropy weight: H = sigma_norm^alpha
    # alpha < 1 makes it more aggressive in exploration
    alpha = 0.7
    entropy_weight = np.power(sigma_norm, alpha)
    
    # Ensure entropy weight has a small floor so we don't completely ignore
    # low-uncertainty regions (where the best solution likely is)
    entropy_weight = np.maximum(entropy_weight, 0.05)
    
    # H-EI: multiplicative combination
    h_ei = ei * entropy_weight
    
    # Add a small UCB component for additional exploration pressure
    # Normalize UCB to be comparable in scale
    ucb = -mu + 0.5 * sigma_safe
    ucb_max = np.max(ucb)
    ucb_min = np.min(ucb)
    if ucb_max - ucb_min > 1e-9:
        ucb_norm = (ucb - ucb_min) / (ucb_max - ucb_min)
    else:
        ucb_norm = np.zeros_like(ucb)
    
    # Normalize H-EI to [0, 1] for blending
    h_ei_max = np.max(h_ei)
    if h_ei_max > 1e-9:
        h_ei_norm = h_ei / h_ei_max
    else:
        h_ei_norm = np.zeros_like(h_ei)
    
    # Combine: primarily H-EI with a small UCB component
    # The multiplicative entropy weighting already provides adaptive exploration,
    # so the UCB component is kept small
    score = 0.9 * h_ei_norm + 0.1 * ucb_norm
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
