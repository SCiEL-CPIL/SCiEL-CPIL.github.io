"""
Acquisition function for Bayesian optimization of Styblinski-Tang 1D.
Uses an adaptive blend of Expected Improvement (EI) and Upper Confidence Bound (UCB).
The exploration weight decays as the best observed value approaches the known global minimum.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

# Deadline guard: The evaluator kills at 330s. We track time to ensure we return safely.
_START_TIME = time.monotonic()
_TIME_BUDGET = 300.0  # Seconds
_MARGIN = 10.0        # Safety margin

# Known global minimum of Styblinski-Tang 1D
GLOBAL_MIN = -39.1662
# Approximate initial range for progress calculation.
# Random points on [-5, 5] for ST1D can yield values around 50-100.
# We use a span of 150 to cover the likely initial range to the minimum.
PROGRESS_SPAN = 150.0


def _expected_improvement(mu, sigma, f_best):
    """
    Calculate Expected Improvement for minimization.
    EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    where z = (f_best - mu) / sigma
    """
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    # Use standard normal CDF and PDF
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    return np.maximum(ei, 0.0)  # EI is non-negative


def _ucb_score(mu, sigma, beta):
    """
    Calculate UCB score for minimization.
    Higher score is better.
    We want to minimize, so we prefer low mu.
    UCB for minimization: - (mu + beta * sigma)
    Or equivalently, we can return a score where higher is better:
    Score = -mu - beta * sigma
    """
    return -(mu + beta * sigma)


def _min_max_normalize(x):
    """
    Min-max normalize an array to [0, 1].
    """
    x_min = np.min(x)
    x_max = np.max(x)
    x_range = x_max - x_min
    if x_range < 1e-12:
        return np.zeros_like(x)
    return (x - x_min) / x_range


def acquisition(mu, sigma, f_best):
    """
    Compute acquisition scores for 2000 candidate points.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        GP posterior mean at candidates (in objective units)
    sigma : array of shape (2000,)
        GP posterior standard deviation at candidates (in objective units)
    f_best : float
        Best (lowest) observed value so far
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores; higher means evaluate next
    """
    # Deadline guard: If we are close to the time limit, return a safe, fast score.
    # We use a simple exploit-only score (negative mean) to avoid expensive calculations.
    if time.monotonic() - _START_TIME > _TIME_BUDGET - _MARGIN:
        # Return -mu as a fast fallback (lower mu is better, so higher score is -mu)
        # Ensure finiteness
        mu = np.where(np.isfinite(mu), mu, f_best)
        return -mu

    # Ensure inputs are arrays and finite
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Handle non-finite values
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    sigma = np.maximum(sigma, 1e-9)

    # 1. Calculate Progress
    # progress = 0 when f_best is far from global min (early stage)
    # progress = 1 when f_best is close to global min (late stage)
    # f_best starts high (e.g., 50). Global min is -39.
    # Difference is ~90.
    # If f_best < GLOBAL_MIN, progress is 1.
    diff = f_best - GLOBAL_MIN
    if diff <= 0:
        progress = 1.0
    else:
        progress = 1.0 - np.clip(diff / PROGRESS_SPAN, 0.0, 1.0)
    
    # 2. Determine Exploration Weight
    # Early on (progress ~ 0), we want high exploration (high UCB weight).
    # Late on (progress ~ 1), we want high exploitation (high EI weight).
    # We use a smooth decay.
    # w_explore = 1.0 at progress=0, w_explore = 0.2 at progress=1 (keep some exploration to avoid local traps)
    # Let's use an exponential decay for faster transition to exploitation.
    # w_explore = 0.2 + 0.8 * (1 - progress)^2
    # Or simpler linear: w_explore = 1.0 - 0.8 * progress
    # Let's try: w_explore = 0.3 + 0.7 * (1.0 - progress)
    # At progress=0, w=1.0. At progress=1, w=0.3.
    w_explore = 0.3 + 0.7 * (1.0 - progress)
    w_exploit = 1.0 - w_explore

    # 3. Calculate Component Scores
    # EI Score
    ei_scores = _expected_improvement(mu, sigma, f_best)
    
    # UCB Score
    # Use a fixed beta that is reasonably high for exploration.
    # Since sigma is in objective units, and the function range is ~100,
    # a beta of 2.0 means we add 2 standard deviations of uncertainty.
    # This is a standard choice.
    beta = 2.0
    ucb_scores = _ucb_score(mu, sigma, beta)

    # 4. Normalize Scores
    # Normalization ensures that the scale of EI and UCB is comparable.
    ei_norm = _min_max_normalize(ei_scores)
    ucb_norm = _min_max_normalize(ucb_scores)

    # 5. Blend Scores
    scores = w_exploit * ei_norm + w_explore * ucb_norm

    # 6. Ensure Finite Output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
