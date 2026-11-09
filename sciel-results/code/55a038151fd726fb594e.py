# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using a combination of Expected Improvement (EI)
    and a tuned exploitation-exploration balance.
    
    For minimization:
    - z = (f_best - mu) / sigma
    - EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    
    We use a weighted combination that emphasizes exploitation when
    improvement is likely, while still allowing some exploration.
    
    The key insight: pure EI can be too greedy or too exploratory depending
    on the problem. We blend EI with a simple exploitation score (negative mu)
    to create a more balanced acquisition.
    """
    # Handle near-zero sigma to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Expected Improvement (for minimization)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Normalize EI to [0, 1] range for blending
    ei_max = np.max(ei)
    if ei_max > 0:
        ei_norm = ei / ei_max
    else:
        ei_norm = ei
    
    # Exploitation score: prefer lower predicted values
    # Scale to be comparable with EI
    mu_min = np.min(mu)
    mu_max = np.max(mu)
    if mu_max > mu_min:
        exploit_score = (mu_max - mu) / (mu_max - mu_min)
    else:
        exploit_score = np.zeros_like(mu)
    
    # Blend: 70% EI, 30% exploitation
    # This creates a balance between finding improvement and exploiting good regions
    alpha = 0.7
    score = alpha * ei_norm + (1 - alpha) * exploit_score
    
    # Add a small penalty for very high uncertainty to avoid pure exploration
    # This is a soft penalty, not a hard constraint
    sigma_penalty = 0.05 * (sigma / np.max(sigma_safe))
    score = score - sigma_penalty
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
