"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining Expected Improvement (EI)
    with a Kriging Believer (KB) exploration term.
    
    The strategy:
    1. Compute standard Expected Improvement (EI) for each candidate.
    2. Compute a "believer" score that approximates the GP mean at the
       point that would be selected by EI, encouraging exploration in
       regions of high uncertainty that might yield large improvements.
    3. Blend EI and KB scores, normalized by cost to respect the
       cost-aware constraint.
    
    Returns: 2000 finite scores (higher = evaluate next).
    """
    # Track elapsed time for deadline guard
    t0 = time.time()
    
    # Ensure sigma is strictly positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Clip EI to be non-negative (numerical stability)
    ei = np.maximum(ei, 0.0)
    
    # Compute Kriging Believer (KB) score:
    # For each candidate, the "believer" prediction is mu + sigma * z*
    # where z* is the standard normal quantile corresponding to the
    # top EI score. We approximate this by using a fixed quantile
    # that balances exploration and exploitation.
    # 
    # A common approach: use the quantile that corresponds to the
    # expected maximum EI across all candidates. We use a simpler
    # approximation: z* = norm.ppf(1 - 1/n) where n is the number
    # of candidates, but this can be extreme. Instead, we use a
    # moderate quantile that encourages exploration without being
    # too aggressive.
    
    # Use a blend: the KB score is mu + alpha * sigma, where alpha
    # is chosen to balance exploration. We use alpha = 1.0 as a
    # reasonable default, which corresponds to the 84th percentile
    # of the standard normal distribution.
    alpha = 1.0
    kb_score = mu + alpha * sigma_safe
    
    # Normalize EI and KB scores to [0, 1] range for blending
    ei_min, ei_max = np.min(ei), np.max(ei)
    ei_range = ei_max - ei_min
    if ei_range < 1e-12:
        ei_norm = np.zeros_like(ei)
    else:
        ei_norm = (ei - ei_min) / ei_range
    
    kb_min, kb_max = np.min(kb_score), np.max(kb_score)
    kb_range = kb_max - kb_min
    if kb_range < 1e-12:
        kb_norm = np.zeros_like(kb_score)
    else:
        kb_norm = (kb_score - kb_min) / kb_range
    
    # Blend EI and KB scores
    # Weight EI more heavily for exploitation, KB for exploration
    w_ei = 0.7
    w_kb = 0.3
    blended = w_ei * ei_norm + w_kb * kb_norm
    
    # Make the score cost-aware: divide by cost
    # cost has shape (2000,) and represents the evaluation cost
    cost_safe = np.maximum(cost, 1e-9)
    scores = blended / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Deadline guard: if we've used too much time, return immediately
    elapsed = time.time() - t0
    if elapsed > 280:  # Safety margin before 300s limit
        return scores
    
    return scores
# EVOLVE-BLOCK-END
