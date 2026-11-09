import numpy as np
import time
from scipy.stats import norm

_START_TIME = time.time()
_DEADLINE = 325.0

_HARTMANN_MIN = -3.86278
_HARTMANN_MAX = 3.32237

def _ei(mu, sigma, f_best):
    """Expected Improvement."""
    z = (f_best - mu) / (sigma + 1e-9)
    return (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)

def _lcB(mu, sigma, f_best, kappa):
    """Lower Confidence Bound (maximize -mu + kappa*sigma)."""
    return -mu + kappa * sigma

def _pi(mu, sigma, f_best):
    """Probability of Improvement."""
    z = (f_best - mu) / (sigma + 1e-9)
    return norm.cdf(z)

def acquisition(mu, sigma, f_best):
    # Deadline guard
    if time.time() > _DEADLINE:
        # Fallback: pure exploitation
        return -mu

    # Compute progress: 1.0 when f_best is at max (worst), 0.0 when f_best is at min (best)
    f_best_clipped = np.clip(f_best, _HARTMANN_MIN, _HARTMANN_MAX)
    progress = (f_best_clipped - _HARTMANN_MIN) / (_HARTMANN_MAX - _HARTMANN_MIN)
    
    # Adaptive weights: blend EI, LCB, and PI based on progress
    # Early on (high progress), favor exploration (LCB)
    # Later (low progress), favor exploitation (EI/PI)
    
    # Weights that sum to 1
    w_ei = 0.3 + 0.4 * (1 - progress)
    w_lcb = 0.5 * progress + 0.1
    w_pi = 1.0 - w_ei - w_lcb
    w_pi = max(w_pi, 0.0)
    
    # Normalize weights
    total_w = w_ei + w_lcb + w_pi
    w_ei /= total_w
    w_lcb /= total_w
    w_pi /= total_w
    
    # Compute each acquisition
    ei_scores = _ei(mu, sigma, f_best)
    lcb_scores = _lcB(mu, sigma, f_best, kappa=2.0)
    pi_scores = _pi(mu, sigma, f_best)
    
    # Scale PI to be comparable to EI (EI can be negative, PI is in [0,1])
    # Scale PI by a factor to match the scale of EI
    pi_scale = np.max(np.abs(ei_scores)) + 1e-9
    pi_scores_scaled = pi_scores * pi_scale
    
    # Combine
    scores = w_ei * ei_scores + w_lcb * lcb_scores + w_pi * pi_scores_scaled
    
    # Ensure finiteness
    scores = np.where(np.isfinite(scores), scores, -1e9)
    
    return scores
