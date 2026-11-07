"""Rewritten acquisition for Bayesian optimization of hartmann 6d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard: keep a safety margin before the 330s wall-clock limit.
_DEADLINE = time.time() + 320.0  # 10s safety margin


def _pi_score(mu, sigma, f_best, eps=1e-4):
    """Probability of Improvement with small exploration jitter."""
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    return norm.cdf(z) + eps * (1.0 - norm.cdf(z))


def _ei_score(mu, sigma, f_best, xi=0.01):
    """Expected Improvement with a small exploitation bias xi."""
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu - xi) / sigma_safe
    ei = (f_best - mu - xi) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    return np.maximum(ei, 0.0)


def acquisition(mu, sigma, f_best):
    """
    Hybrid acquisition: blends Probability of Improvement (PI) with
    Expected Improvement (EI) to balance exploitation and exploration.
    
    PI prioritizes points with high probability of beating f_best,
    avoiding heavy-tailed exploration that wastes evaluations in 6D.
    EI provides a magnitude signal for how much improvement is expected.
    
    The blend uses a normalized combination:
      score = w * PI_normalized + (1-w) * EI_normalized
    where w is chosen to favor PI slightly (w=0.6) for robustness.
    """
    # Check deadline
    if time.time() > _DEADLINE:
        # Return a safe fallback: just PI scores
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        return norm.cdf(z)
    
    # Compute PI scores
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    pi_scores = norm.cdf(z)
    
    # Compute EI scores with small xi for exploitation bias
    xi = 0.01
    z_ei = (f_best - mu - xi) / sigma_safe
    ei_scores = np.maximum((f_best - mu - xi) * norm.cdf(z_ei) + sigma_safe * norm.pdf(z_ei), 0.0)
    
    # Normalize both to [0, 1] for blending
    pi_min, pi_max = pi_scores.min(), pi_scores.max()
    ei_min, ei_max = ei_scores.min(), ei_scores.max()
    
    pi_range = pi_max - pi_min
    ei_range = ei_max - ei_min
    
    if pi_range < 1e-12:
        pi_norm = np.ones_like(pi_scores)
    else:
        pi_norm = (pi_scores - pi_min) / pi_range
    
    if ei_range < 1e-12:
        ei_norm = np.ones_like(ei_scores)
    else:
        ei_norm = (ei_scores - ei_min) / ei_range
    
    # Blend: favor PI for robustness in high-dim, but keep EI signal
    w = 0.6
    scores = w * pi_norm + (1.0 - w) * ei_norm
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
