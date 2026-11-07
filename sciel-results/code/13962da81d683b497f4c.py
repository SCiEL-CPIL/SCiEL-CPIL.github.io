"""Rewritten program: Thompson Sampling acquisition for cost-aware BO."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Track wall-clock deadline (330s limit, use 310s margin)
_T0 = time.time()
_DEADLINE = 310.0


def _time_ok():
    return (time.time() - _T0) < _DEADLINE


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition using Thompson Sampling for Bayesian optimization
    of Styblinski-Tang 2D.
    
    mu, sigma, cost: arrays of shape (2000,)
    f_best: scalar, lowest observed value
    Returns: 2000 finite scores (higher = evaluate next)
    """
    # Deadline guard
    if not _time_ok():
        mu = np.asarray(mu, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        cost = np.asarray(cost, dtype=np.float64)
        sigma_safe = np.maximum(sigma, 1e-8)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        score = ei / np.maximum(cost, 1e-6)
        return np.where(np.isfinite(score), score, 0.0)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Safety: ensure finite and positive
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)

    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-6)

    f_best_val = float(f_best)
    if not np.isfinite(f_best_val):
        f_best_val = 0.0

    # Thompson Sampling: draw a sample from the GP posterior
    # sample = mu + sigma * z, where z ~ N(0, 1)
    # We want to maximize (f_best - sample) / cost
    # This is equivalent to minimizing sample / cost, or maximizing improvement potential
    
    # Use a single random draw for all 2000 candidates (consistent within one acquisition call)
    # This is standard Thompson Sampling: one posterior sample per candidate set
    rng = np.random.default_rng()
    z_samples = rng.standard_normal(2000)
    
    # Posterior predictive sample
    sample = mu + sigma_safe * z_samples
    
    # Score: expected improvement relative to cost
    # Higher score = more promising to evaluate
    # (f_best - sample) / cost gives the cost-normalized potential improvement
    # If sample < f_best, this is positive (potential improvement)
    # If sample > f_best, this is negative (worse than current best)
    
    improvement = f_best_val - sample
    score = improvement / cost_safe
    
    # Add a small exploration component to avoid completely ignoring
    # high-uncertainty regions that might have hidden minima
    # The Thompson Sampling sample already incorporates uncertainty through z_samples,
    # but we can add a mild bonus for very high uncertainty regions
    
    sigma_max = np.max(sigma_safe) if np.any(sigma_safe > 1e-8) else 1.0
    if sigma_max > 1e-8:
        # Mild uncertainty bonus, scaled down to not dominate
        explore_bonus = 0.1 * (sigma_safe / sigma_max)
        score = score + explore_bonus
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
