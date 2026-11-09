# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global timer for safety
_START_TIME = time.time()
_TIME_LIMIT = 300  # Seconds

def _check_time():
    """Check if we are close to the time limit."""
    return (time.time() - _START_TIME) > (_TIME_LIMIT - 5)

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using a hybrid of Expected Improvement (EI)
    and Upper Confidence Bound (UCB) with adaptive exploration.
    
    For minimization:
    - EI = (f_best - mu) * Phi(z) + sigma * phi(z) where z = (f_best - mu) / sigma
    - UCB = -mu + beta * sigma
    - We blend these two to balance exploitation (EI) and exploration (UCB)
    
    The key insight: EI is good for exploitation but can be too greedy.
    UCB provides a linear exploration bonus that helps escape local optima.
    By blending them, we get a more robust acquisition function.
    
    We also adapt the exploration rate based on the number of evaluations
    estimated from the sigma distribution.
    """
    # Safety check: if we are running out of time, return a safe fallback
    if _check_time():
        # Fallback: return -mu (pure exploitation) to ensure we finish
        return -mu
    
    # Handle near-zero sigma to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Calculate z-score for EI
    z = (f_best - mu) / sigma_safe
    
    # Calculate Expected Improvement (for minimization)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Normalize EI to [0, 1] range for blending
    ei_max = np.max(ei)
    if ei_max > 0:
        ei_norm = ei / ei_max
    else:
        ei_norm = ei
    
    # Estimate number of evaluations so far
    # Points with very small sigma are likely near previously evaluated points
    sigma_threshold = 1e-3
    n_low_sigma = np.sum(sigma < sigma_threshold)
    # Base offset of 5 (initial points) plus estimated additional evaluations
    n_evals_est = max(5, 5 + int(n_low_sigma / 50))
    
    # Dynamic beta for UCB: starts higher for exploration, decreases as we learn more
    # Using a form that transitions from exploration to exploitation
    beta = 0.5 + 0.3 * np.sqrt(n_evals_est)
    
    # UCB for minimization: we want to minimize mu - beta*sigma
    # So the UCB score (higher is better) is: -(mu - beta*sigma) = -mu + beta*sigma
    ucb_score = -mu + beta * sigma_safe
    
    # Normalize UCB score to [0, 1] range for blending
    ucb_min = np.min(ucb_score)
    ucb_max = np.max(ucb_score)
    if ucb_max > ucb_min:
        ucb_norm = (ucb_score - ucb_min) / (ucb_max - ucb_min)
    else:
        ucb_norm = np.zeros_like(ucb_score)
    
    # Blend: 60% EI, 40% UCB
    # This creates a balance between finding improvement and exploring uncertain regions
    alpha = 0.6
    score = alpha * ei_norm + (1 - alpha) * ucb_norm
    
    # Add a small penalty for very high uncertainty to avoid pure exploration
    sigma_penalty = 0.03 * (sigma / np.max(sigma_safe))
    score = score - sigma_penalty
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
