import numpy as np
from scipy.stats import norm
import time

# Global time tracking for deadline guard
_start_time = time.time()
_DEADLINE_OFFSET = 320.0  # Safety margin before 330s kill


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Quantile of the Posterior (QoP) acquisition function.
    
    For minimization, we target the p-th quantile of the posterior:
    q_p(x) = mu(x) + z_p * sigma(x), where z_p is the p-th percentile z-score.
    Lower quantiles (small p) are more conservative/exploitative.
    
    We adaptively choose p based on progress: early on use low p (exploratory),
    later use higher p (exploitative). Score = -q_p / cost.
    
    Parameters:
    mu: Array of shape (2000,) - GP mean
    sigma: Array of shape (2000,) - GP standard deviation
    f_best: Float - Best observed value
    cost: Array of shape (2000,) - Cost of evaluation
    
    Returns:
    Array of shape (2000,) - Acquisition scores (higher is better)
    """
    # Deadline guard
    if time.time() - _start_time > _DEADLINE_OFFSET:
        # Fallback: simple cost-normalized exploit
        mu_safe = np.asarray(mu)
        cost_safe = np.asarray(cost)
        cost_safe = np.maximum(cost_safe, 1e-9)
        return -mu_safe / cost_safe

    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    cost = np.asarray(cost)
    
    # Ensure sigma and cost are positive to avoid division errors
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Adaptive quantile level: 
    # Early (f_best high ~20-40): use low quantile (e.g., 5th percentile) for exploration
    # Late (f_best low ~0-5): use higher quantile (e.g., 50th percentile) for exploitation
    f_best_clamped = np.clip(f_best, 0.0, 50.0)
    progress = 1.0 - (f_best_clamped / 50.0)  # 0 when f_best=50, 1 when f_best=0
    
    # Quantile p: interpolate from 0.05 (exploratory) to 0.5 (exploitative)
    p = 0.05 + 0.45 * progress
    
    # z-score for the p-th percentile
    z_p = norm.ppf(p)
    
    # Quantile of the posterior: q_p = mu + z_p * sigma
    # For minimization, lower q_p is better
    q_p = mu + z_p * sigma_safe
    
    # Cost-aware score: higher is better, so negate the quantile and divide by cost
    score = -q_p / cost_safe
    
    # Ensure finite scores
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
