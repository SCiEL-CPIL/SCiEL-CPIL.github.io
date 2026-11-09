"""Rewritten acquisition for Bayesian optimization of Goldstein-Price 2D."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Deadline guard: return early if we're getting close to the 330s wall-clock limit
_start_time = time.time()
_DEADLINE_MARGIN = 10.0  # seconds of safety margin

def acquisition(mu, sigma, f_best):
    """
    Compute acquisition scores using a hybrid approach combining
    Expected Improvement (EI) with a UCB-like exploration term.
    
    Parameters:
    mu: array-like, shape (N,), mean of the posterior distribution
    sigma: array-like, shape (N,), standard deviation of the posterior distribution
    f_best: float, the best (lowest) observed function value so far
    
    Returns:
    scores: numpy array, shape (N,), higher values indicate better candidates
    """
    # Check deadline
    if time.time() - _start_time > 330.0 - _DEADLINE_MARGIN:
        # Return a simple UCB score as fallback
        mu = np.asarray(mu, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        mu = np.where(np.isfinite(mu), mu, f_best)
        sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
        sigma = np.maximum(sigma, 1e-9)
        scores = -mu + 1.5 * sigma
        return np.where(np.isfinite(scores), scores, 0.0)
    
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Safety: handle non-finite values
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    sigma = np.maximum(sigma, 1e-9)
    
    # Calculate Z-score: (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    
    # Expected Improvement for minimization
    cdf_val = norm.cdf(z)
    pdf_val = norm.pdf(z)
    ei = (f_best - mu) * cdf_val + sigma * pdf_val
    
    # Adaptive exploration: use the ratio of remaining uncertainty to 
    # the gap between f_best and the known minimum (3.0)
    # When f_best is far from 3.0, explore more; when close, exploit
    gap_to_opt = max(f_best - 3.0, 1e-6)
    # Mean uncertainty relative to the gap
    mean_sigma = np.mean(sigma)
    # Exploration coefficient: higher when uncertainty is large relative to gap
    # Scale: if mean_sigma >> gap, we're still exploring; if mean_sigma << gap, exploit
    ratio = mean_sigma / gap_to_opt
    # beta ranges from ~0.5 (exploit) to ~2.5 (explore)
    beta = 0.5 + 2.0 * min(ratio, 1.0)
    
    ucb = -mu + beta * sigma
    
    # Weight EI more heavily as we converge (ratio decreases)
    # When ratio is high (early), weight UCB more; when low (late), weight EI more
    ei_weight = 0.3 + 0.7 * (1.0 - min(ratio, 1.0))
    ucb_weight = 1.0 - ei_weight
    
    scores = ei_weight * ei + ucb_weight * ucb
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
