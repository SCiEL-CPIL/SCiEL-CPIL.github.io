"""Acquisition function for Bayesian optimization of Beale 2D.

Uses a modified Expected Improvement (EI) acquisition function.
The score is a combination of standard EI and an uncertainty-weighted
exploration term. The exploration term is modulated by the likelihood
of improvement (via the standard normal PDF of the z-score), which
encourages exploration in regions that are both uncertain and likely
to contain improvements over f_best.
"""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def _expected_improvement(mu, sigma, f_best):
    """Standard Expected Improvement for minimization."""
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    z = (f_best - mu) / sigma_safe
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    # Handle edge cases where sigma is very small
    ei = np.where(sigma < eps, np.maximum(f_best - mu, 0.0), ei)
    return np.where(np.isfinite(ei), ei, 0.0)


def acquisition(mu, sigma, f_best):
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    n = mu.shape[0]
    if n == 0:
        return np.array([], dtype=float)

    # Numerical safety
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-10)
    mu = np.where(np.isfinite(mu), mu, f_best)
    f_best = float(f_best)
    
    # Core EI score
    ei = _expected_improvement(mu, sigma, f_best)

    # Calculate Z-scores for exploration modulation
    z = (f_best - mu) / sigma
    
    # Exploration term:
    # We want to explore where sigma is high AND z is reasonable (not too negative, i.e., mu isn't already much better than f_best).
    # norm.pdf(z) is high when z is close to 0.
    # This effectively creates a "sweet spot" for exploration: uncertain regions near the current best.
    # We scale this by sigma to make it in objective units.
    
    # To make it adaptive:
    # If f_best is still very high (early), we might want more global exploration.
    # If f_best is close to 0 (late), we want focused exploitation.
    
    # Simple adaptive coefficient:
    # As f_best -> 0, we reduce exploration.
    # Assuming Beale's range is roughly [0, 200], f_best < 1 is a good sign.
    # Let's use a soft decay.
    # decay = 1.0 / (1.0 + f_best) is not quite right if f_best starts at 100.
    # Let's use: decay = np.exp(-0.5 * f_best) ? No, f_best can be > 10.
    
    # Better: Use the mean sigma to scale the exploration.
    # If sigma is generally small, exploration is less needed.
    sigma_mean = float(np.mean(sigma))
    
    # Exploration weight
    # Base weight is small.
    base_weight = 0.25
    
    # Modulate by the "promise" of the region.
    # norm.pdf(z) peaks at z=0.
    # We multiply by sigma to get units of f.
    exploration_term = base_weight * sigma * norm.pdf(z)
    
    # Add a small pure uncertainty term to avoid ignoring high-sigma low-z areas?
    # No, the pdf handles that.
    
    # Combine
    score = ei + exploration_term
    
    # Safety
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
