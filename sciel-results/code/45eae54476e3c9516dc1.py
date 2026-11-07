"""
Acquisition function for Bayesian optimization of Styblinski-Tang 1D.
Uses Expected Improvement (EI) with an adaptive UCB exploration bonus.
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
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    return np.maximum(ei, 0.0)


def _ucb_score(mu, sigma, f_best, kappa):
    """
    Upper Confidence Bound for minimization.
    UCB = mu - kappa * sigma
    Higher score means more promising.
    """
    return -mu + kappa * sigma


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
    diff = f_best - GLOBAL_MIN
    if diff <= 0:
        progress = 1.0
    else:
        progress = 1.0 - np.clip(diff / PROGRESS_SPAN, 0.0, 1.0)
    
    # 2. Compute Expected Improvement
    ei = _expected_improvement(mu, sigma, f_best)
    
    # 3. Compute UCB score for exploration
    # kappa adapts: higher in early stages (more exploration), lower in late stages
    kappa = 0.3 + 1.2 * (1.0 - progress)
    ucb = _ucb_score(mu, sigma, f_best, kappa)
    
    # 4. Blend EI and UCB
    # Normalize both to [0,1] for fair blending
    ei_norm = _min_max_normalize(ei)
    ucb_norm = _min_max_normalize(ucb)
    
    # Weight: more UCB in early stages, more EI in late stages
    w_ucb = 0.3 + 0.5 * (1.0 - progress)
    w_ei = 1.0 - w_ucb
    
    scores = w_ei * ei_norm + w_ucb * ucb_norm
    
    # 5. Ensure Finite Output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
