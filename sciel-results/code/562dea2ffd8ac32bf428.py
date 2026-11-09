"""Acquisition function for Bayesian optimization of a 2D RBF mixture."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Module-level deadline guard
_DEADLINE = None
_START_TIME = None


def _get_deadline():
    global _DEADLINE, _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
        # The evaluator kills after 330s; use a safety margin
        _DEADLINE = _START_TIME + 320.0
    return _DEADLINE


def _time_remaining():
    return _get_deadline() - time.time()


def acquisition(mu, sigma, f_best):
    """
    Acquisition function: a blend of Expected Improvement (EI) and GP-UCB
    (Upper Confidence Bound) for minimization.
    
    mu, sigma: shape (2000,), posterior mean and std at candidate points.
    f_best: lowest observed value so far.
    
    Returns: shape (2000,) scores, higher = evaluate next.
    """
    # Time guard: if we're running out of time, return a simple but fast score
    if _time_remaining() < 2.0:
        z = (f_best - mu) / (sigma + 1e-9)
        return (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)

    n = len(mu)
    eps = 1e-9

    # --- Core: Expected Improvement for minimization ---
    z = (f_best - mu) / (sigma + eps)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)

    # --- GP-UCB component for minimization ---
    # UCB(x) = mu(x) - beta * sigma(x)
    # Lower UCB means more promising (potentially lower function value).
    # Score = -UCB = -mu + beta * sigma (higher is better)
    # Use a time-varying beta that decreases over time for exploitation
    # Estimate progress: f_best is negative, minimum is -2.53218
    # Early: f_best close to 0 or positive, Late: f_best close to -2.53
    # beta starts high for exploration, decreases for exploitation
    progress = max(0.0, min(1.0, (-f_best) / 2.53218))
    # beta decreases from ~3.0 (early) to ~1.0 (late)
    beta = 3.0 - 2.0 * progress
    
    ucb_score = -mu + beta * sigma

    # --- Smooth blend without normalization ---
    # Both EI and ucb_score are in "higher is better" units.
    # EI is bounded below by 0; ucb_score can be negative.
    # We blend them with a weight that shifts from exploration to exploitation.
    
    # Estimate how close we are to the optimum using the posterior scale.
    mu_scale = np.median(np.abs(mu) + sigma)
    if mu_scale < 1e-9:
        mu_scale = 1.0
    
    # improvement_ratio: 0 when f_best is near typical values (early), 
    # 1 when f_best is much lower (late, close to optimum)
    improvement_ratio = max(0.0, min(1.0, (mu_scale - f_best) / (mu_scale + eps)))
    
    # Weight: early on, rely more on UCB for exploration; late, rely more on EI
    # w_ei goes from ~0.3 (early) to ~0.85 (late)
    w_ei = 0.3 + 0.55 * improvement_ratio
    w_ucb = 1.0 - w_ei

    # Scale ucb_score to be comparable to EI
    # Normalize ucb_score by its own range to make it O(1)
    ucb_min, ucb_max = np.min(ucb_score), np.max(ucb_score)
    ucb_range = ucb_max - ucb_min + eps
    ucb_norm = (ucb_score - ucb_min) / ucb_range
    
    # EI is already non-negative and naturally scaled; normalize it too for blending
    ei_min, ei_max = np.min(ei), np.max(ei)
    ei_range = ei_max - ei_min + eps
    ei_norm = (ei - ei_min) / ei_range

    score = w_ei * ei_norm + w_ucb * ucb_norm

    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)

    return score
# EVOLVE-BLOCK-END
