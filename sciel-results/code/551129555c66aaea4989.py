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
    Acquisition function: a blend of Expected Improvement (EI) and a
    knowledge-gradient-inspired correction that accounts for the
    probability that a candidate is the global minimum.
    
    mu, sigma: shape (2000,), posterior mean and std at candidate points.
    f_best: lowest observed value so far.
    
    Returns: shape (2000,) scores, higher = evaluate next.
    """
    # Time guard: if we're running out of time, return a simple but fast score
    if _time_remaining() < 2.0:
        # Fallback: simple EI with epsilon
        z = (f_best - mu) / (sigma + 1e-9)
        return (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)

    n = len(mu)
    eps = 1e-9

    # --- Core: Expected Improvement ---
    # Standard EI for minimization: EI(x) = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    z = (f_best - mu) / (sigma + eps)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)

    # --- Structural enhancement: blend with a UCB-like exploration term ---
    # The idea: pure EI can be too greedy early on. We add a small exploration
    # bonus that is proportional to sigma, but modulated by how far mu is
    # from f_best. This is different from vanilla UCB because the beta
    # coefficient adapts to the current state of the search.
    
    # Adaptive beta: larger when we've barely beaten random (f_best close to 
    # the typical scale of the function), smaller as we converge.
    # We estimate the function scale from the current posterior.
    # A robust scale estimate: median of |mu| + sigma
    scale = np.median(np.abs(mu) + sigma)
    if scale < 1e-9:
        scale = 1.0
    
    # beta decays as f_best improves relative to scale
    # When f_best is much smaller than typical values, we're close to optimum -> less exploration
    # When f_best is near typical values, we're early -> more exploration
    improvement_ratio = max(0.0, (scale - f_best) / (scale + eps))
    # beta ranges from ~2.0 (early) to ~0.5 (late)
    beta = 0.5 + 1.5 * (1.0 - improvement_ratio)

    # UCB component
    ucb = mu + beta * sigma

    # We want to MINIMIZE, so lower ucb is better.
    # Convert to a "score" where higher is better: negate ucb.
    # But we need to blend with EI which is already a "higher is better" score.
    
    # Normalize both to [0, 1] range for blending
    ei_min, ei_max = np.min(ei), np.max(ei)
    ei_range = ei_max - ei_min + eps
    ei_norm = (ei - ei_min) / ei_range

    ucb_neg = -ucb
    ucb_neg_min, ucb_neg_max = np.min(ucb_neg), np.max(ucb_neg)
    ucb_neg_range = ucb_neg_max - ucb_neg_min + eps
    ucb_norm = (ucb_neg - ucb_neg_min) / ucb_neg_range

    # Blend: weight EI more heavily as we progress
    # Early: 50% EI + 50% UCB. Late: 80% EI + 20% UCB.
    w_ei = 0.5 + 0.3 * improvement_ratio  # 0.5 to 0.8
    w_ucb = 1.0 - w_ei

    score = w_ei * ei_norm + w_ucb * ucb_norm

    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)

    return score
# EVOLVE-BLOCK-END
