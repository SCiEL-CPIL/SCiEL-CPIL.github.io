import numpy as np
import time
from scipy.stats import norm

_T0 = time.time()
_DEADLINE = 320.0
_EPS = 1e-12


def _expected_improvement(mu, sigma, f_best):
    """Compute Expected Improvement for minimization."""
    sigma_safe = np.maximum(sigma, _EPS)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    return np.maximum(ei, 0.0)


def _normalize(arr):
    """Min-max normalize to [0, 1]."""
    mn = np.min(arr)
    mx = np.max(arr)
    if mx - mn < _EPS:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


def _get_progress(eval_count, total_steps=25):
    """Get progress in [0, 1] over the 25 BO evaluation steps."""
    if eval_count <= 0:
        return 0.0
    return min(eval_count / float(total_steps), 1.0)


# Track evaluation count
_eval_count = 0


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using a Kriging-Believer hybrid with EI anchoring.
    
    Core idea: construct a single "believed objective" score that blends
    exploitation (low predicted mean) and exploration (high uncertainty)
    via an adaptive quantile of the posterior. Early on, we believe in
    a more pessimistic (lower) quantile to encourage exploration; later
    we shift toward the mean for exploitation. We also add a small EI
    term to anchor the search around the incumbent.
    
    Args:
        mu: array of shape (2000,) - posterior mean
        sigma: array of shape (2000,) - posterior standard deviation
        f_best: float - lowest observed value
    
    Returns:
        array of shape (2000,) - scores, higher means evaluate next
    """
    global _eval_count
    
    # Deadline guard: fall back to pure EI
    if time.time() - _T0 > _DEADLINE:
        sigma_safe = np.maximum(sigma, _EPS)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        return np.maximum(ei, 0.0)
    
    # Increment evaluation counter
    _eval_count += 1
    
    sigma_safe = np.maximum(sigma, _EPS)
    progress = _get_progress(_eval_count)
    
    # --- Kriging Believer: pick an adaptive quantile of the posterior ---
    # Early: use a low quantile (e.g., 5th percentile) → explore uncertain regions
    # Late:  use the mean (50th percentile) → exploit
    # Quantile parameter q in [0, 1]; score = mu + z_q * sigma where z_q = Phi^{-1}(q)
    # For minimization, we want to minimize mu + z_q * sigma, so we maximize -(mu + z_q*sigma)
    # Early q ≈ 0.05 (z_q ≈ -1.645), late q ≈ 0.5 (z_q ≈ 0)
    q = 0.05 + 0.45 * progress  # 0.05 → 0.50
    # Use norm.ppf once per call (scalar)
    z_q = norm.ppf(q)  # scalar, e.g., -1.645 early, 0.0 late
    
    # Believed objective (lower is better): mu + z_q * sigma
    # We want to MINIMIZE this, so score = -(mu + z_q * sigma) = -mu - z_q * sigma
    # Since z_q is negative early, -z_q is positive → adds exploration
    believer_score = -mu - z_q * sigma_safe  # higher = more promising
    
    # --- EI component for anchoring ---
    ei = _expected_improvement(mu, sigma_safe, f_best)
    
    # --- Normalize both to comparable scale ---
    believer_norm = _normalize(believer_score)
    ei_norm = _normalize(ei)
    
    # --- Blend: shift from believer (exploration) to EI (exploitation) ---
    # Early: 80% believer, 20% EI
    # Late:  40% believer, 60% EI
    w_believer = 0.8 - 0.4 * progress
    w_ei = 1.0 - w_believer
    
    score = w_believer * believer_norm + w_ei * ei_norm
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
