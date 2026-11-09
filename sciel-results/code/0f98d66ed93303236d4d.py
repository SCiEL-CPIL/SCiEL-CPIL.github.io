"""Cost-aware acquisition for Bayesian optimization of Hartmann-3D."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

# Global deadline guard
_DEADLINE_START = time.time()
_TIME_LIMIT = 325.0  # Safety margin before 330s kill


def _time_left():
    return _TIME_LIMIT - (time.time() - _DEADLINE_START)


def _min_max_normalize(arr):
    """Normalize array to [0, 1] using min-max scaling."""
    arr = np.asarray(arr, dtype=np.float64)
    mn = np.min(arr)
    mx = np.max(arr)
    if mx - mn < 1e-9:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using min-max normalized EI and dynamic UCB.
    
    Parameters:
    -----------
    mu : array of shape (2000,) - GP mean predictions
    sigma : array of shape (2000,) - GP standard deviation predictions
    f_best : float - best (lowest) observed objective value
    cost : array of shape (2000,) - evaluation costs for each candidate
    
    Returns:
    --------
    scores : array of shape (2000,) - acquisition scores (higher = evaluate next)
    """
    # Check deadline
    if _time_left() < 0.5:
        return np.zeros_like(mu)
    
    # Ensure finite values
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-10)
    
    # === Expected Improvement (exploitation) ===
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # === Dynamic UCB for minimization (exploration) ===
    # score_ucb = kappa * sigma - mu (higher = more attractive for minimization)
    mu_range = np.ptp(mu)
    if mu_range < 1e-9:
        mu_range = 1.0
    min_mu = np.min(mu)
    gap = f_best - min_mu
    
    # Dynamic kappa: more exploration when gap is large
    kappa_base = 1.5
    if gap > 1e-6:
        kappa = kappa_base + 1.0 * np.tanh(gap / (0.5 * mu_range + 1e-9))
    else:
        kappa = kappa_base * 0.5
    kappa = np.clip(kappa, 0.5, 4.0)
    
    # UCB raw score (before cost adjustment)
    ucb_raw = kappa * sigma_safe - mu
    
    # === Min-max normalize both components ===
    ei_norm = _min_max_normalize(ei)
    ucb_norm = _min_max_normalize(ucb_raw)
    
    # === Adaptive blend: shift from exploration to exploitation as we converge ===
    # When gap is small (near optimum), weight more toward EI (exploitation)
    # When gap is large, weight more toward UCB (exploration)
    if gap > 1e-6 and mu_range > 1e-9:
        gap_frac = np.clip(gap / (2.0 * mu_range), 0.0, 1.0)
    else:
        gap_frac = 0.0
    
    # alpha for EI, beta for UCB
    alpha = 0.6 + 0.3 * (1.0 - gap_frac)  # 0.6 to 0.9
    beta = 1.0 - alpha                      # 0.4 to 0.1
    
    blended = alpha * ei_norm + beta * ucb_norm
    
    # === Cost-aware adjustment ===
    scores = blended / cost_safe
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
