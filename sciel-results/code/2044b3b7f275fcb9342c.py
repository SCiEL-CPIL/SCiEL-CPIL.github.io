# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining Expected Improvement (EI)
    and Upper Confidence Bound (UCB) with z-score normalization.
    
    Uses z-score normalization to balance exploitation and exploration,
    with a fixed 75/25 ratio favoring exploitation.
    """
    # Deadline guard
    if not hasattr(acquisition, '_start_time'):
        acquisition._start_time = time.time()
    
    elapsed = time.time() - acquisition._start_time
    if elapsed > 325.0:
        mu_fallback = np.asarray(mu, dtype=np.float64)
        cost_fallback = np.asarray(cost, dtype=np.float64)
        score = -mu_fallback / np.maximum(cost_fallback, 1e-9)
        return np.where(np.isfinite(score), score, 0.0)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    sigma_safe = np.maximum(sigma, 1e-12)
    cost_safe = np.maximum(cost, 1e-9)

    # Expected Improvement for minimization
    z = (f_best - mu) / sigma_safe
    z_clipped = np.clip(z, -50.0, 50.0)
    
    cdf_z = norm.cdf(z_clipped)
    pdf_z = norm.pdf(z_clipped)
    
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    ei = np.maximum(ei, 0.0)
    
    score_ei = ei / cost_safe

    # Upper Confidence Bound for minimization
    beta = 2.0
    ucb_score = -mu + beta * sigma_safe
    score_ucb = ucb_score / cost_safe

    # Z-score normalization
    def zscore(arr):
        m = np.mean(arr)
        s = np.std(arr)
        if s < 1e-9:
            return np.zeros_like(arr)
        return (arr - m) / s
        
    z_ei = zscore(score_ei)
    z_ucb = zscore(score_ucb)
    
    # Combine with 75% exploitation, 25% exploration
    final_score = 0.75 * z_ei + 0.25 * z_ucb
    
    final_score = np.where(np.isfinite(final_score), final_score, 0.0)
    
    return final_score
# EVOLVE-BLOCK-END
