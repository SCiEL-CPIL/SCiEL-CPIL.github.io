import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a blend of Expected Improvement (EI)
    and Probability of Improvement (PI), with robust normalization and adaptive
    exploration.
    
    mu, sigma, f_best, cost: arrays of shape (2000,)
    Returns: 2000 finite scores (higher = evaluate next)
    """
    t_start = time.time()
    
    # Ensure inputs are properly shaped and finite
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    f_best = float(f_best)
    
    # Replace non-finite values
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    
    # Ensure sigma is positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-10)
    
    # Compute z-score for EI
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (stable computation)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Probability of Improvement
    pi = norm.cdf(z)
    
    # Normalize EI and PI to [0, 1] range using max-based normalization
    # This is more robust than min-max when values are concentrated
    ei_max = np.max(ei) if np.max(ei) > 0 else 1.0
    pi_max = np.max(pi) if np.max(pi) > 0 else 1.0
    
    ei_norm = ei / ei_max
    pi_norm = pi / pi_max
    
    # Adaptive blending: weight EI more heavily when it provides clear signal,
    # but maintain PI contribution for exploration in uncertain regions
    # When EI is high relative to its max, exploit; when low, explore via PI
    ei_relative = ei_norm
    # Blend with slight emphasis on EI for exploitation, PI for exploration
    blended = 0.7 * ei_norm + 0.3 * pi_norm
    
    # Cost-aware: divide by cost (higher cost = lower score)
    scores = blended / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check time - if we're getting close to the limit, return immediately
    t_elapsed = time.time() - t_start
    if t_elapsed > 280:
        return scores
    
    return scores
