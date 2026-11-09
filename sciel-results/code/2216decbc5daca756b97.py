import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = 295.0  # safety margin before 300s kill (330s total with overhead)

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware UCB acquisition function for Bayesian optimization of hartmann 3d.
    
    Strategy:
    1. Compute a cost-aware Upper Confidence Bound: score = (mu + kappa * sigma) / cost
    2. Dynamically tune kappa based on the spread of sigma and remaining budget proxy.
    3. Blend with a small PI component to ensure exploration in uncertain regions.
    4. Use min-max normalization to keep scores in a stable range.
    """
    # Deadline guard
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE - 5.0:
        # Return a simple fallback: negative mean divided by cost (exploit known best)
        return -mu / np.maximum(cost, 1e-9)
    
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Guard against zero or negative sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    sigma_safe = np.where(np.isfinite(sigma_safe), sigma_safe, 1e-9)
    
    # Ensure cost is positive and finite
    cost_safe = np.maximum(cost, 1e-9)
    cost_safe = np.where(np.isfinite(cost_safe), cost_safe, 1.0)
    
    # Ensure mu is finite
    mu = np.where(np.isfinite(mu), mu, f_best)
    
    # Compute z-score for improvement
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -10, 10)
    
    # Probability of Improvement (for blending)
    pi = norm.cdf(z)
    
    # Compute UCB score: (mu + kappa * sigma) / cost
    # Since we are minimizing, we want to maximize the score.
    # Lower mu is better, higher sigma is better for exploration.
    # The UCB for minimization is typically: mu - kappa * sigma
    # But since we want higher scores to be better, we can use:
    # score = -(mu - kappa * sigma) / cost = (kappa * sigma - mu) / cost
    
    # Dynamically tune kappa:
    # Higher sigma spread suggests more exploration is needed.
    # Also, as we have more data (higher cost spent), we might explore less.
    # Here we use the mean normalized sigma as a proxy for exploration need.
    sigma_min = np.min(sigma_safe)
    sigma_max = np.max(sigma_safe)
    sigma_range = sigma_max - sigma_min
    if sigma_range < 1e-12:
        sigma_norm = np.zeros_like(sigma_safe)
    else:
        sigma_norm = (sigma_safe - sigma_min) / sigma_range
    
    # Base kappa is moderate, increased by average uncertainty
    # The average normalized sigma is in [0, 1]
    mean_sigma_norm = np.mean(sigma_norm)
    # Kappa ranges from ~0.5 to ~1.5
    kappa = 0.5 + 1.0 * mean_sigma_norm
    
    # UCB score (for minimization, higher is better):
    # We want to favor low mu and high sigma.
    # ucb_score = (kappa * sigma - mu) / cost
    ucb_score = (kappa * sigma_safe - mu) / cost_safe
    
    # Normalize UCB score to [0, 1]
    ucb_min = np.min(ucb_score)
    ucb_max = np.max(ucb_score)
    ucb_range = ucb_max - ucb_min
    if ucb_range < 1e-12:
        ucb_norm = np.zeros_like(ucb_score)
    else:
        ucb_norm = (ucb_score - ucb_min) / ucb_range
    
    # Normalize PI to [0, 1]
    pi_min = np.min(pi)
    pi_max = np.max(pi)
    pi_range = pi_max - pi_min
    if pi_range < 1e-12:
        pi_norm = np.zeros_like(pi)
    else:
        pi_norm = (pi - pi_min) / pi_range
    
    # Blend: primarily UCB, with a small PI component for robustness
    # The UCB already includes exploration via sigma, so PI weight is small
    blended = 0.85 * ucb_norm + 0.15 * pi_norm
    
    # Ensure all scores are finite
    blended = np.where(np.isfinite(blended), blended, 0.0)
    
    return blended
