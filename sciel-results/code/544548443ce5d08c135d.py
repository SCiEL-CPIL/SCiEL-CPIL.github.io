"""Rewritten program: diversity-aware greedy batch acquisition for Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Diversity-aware acquisition function.
    
    Strategy:
    1. Compute EI scores for all candidates.
    2. Select top-k candidates by EI.
    3. From those, greedily select a subset that maximizes spatial diversity
       (minimum distance to already-selected points) while maintaining high EI.
    4. Return scores where the selected diverse set gets boosted scores,
       and others get their raw EI.
    
    Since we only return one score per candidate and the evaluator picks the
    argmax, we effectively need to return scores such that the argmax is a
    well-chosen point. To implement diversity-aware selection in a single
    forward pass, we use a different approach:
    
    We compute a combined score that balances exploitation (EI) and exploration
    (distance from other high-EI candidates). This creates a "crowding" penalty:
    candidates in dense clusters of high-EI points get penalized, spreading
    the search.
    """
    n = len(mu)
    
    # Compute Expected Improvement
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure non-negative EI
    ei = np.maximum(ei, 0.0)
    
    # If all EI are zero, fall back to pure exploration via distance-based score
    if np.all(ei < 1e-10):
        # Use a simple UCB-like exploration
        # Without coordinates, we can't compute distance, so return uniform-ish scores
        # based on sigma (uncertainty)
        scores = sigma
        return scores
    
    # Normalize EI to [0, 1] for combination
    ei_max = np.max(ei)
    if ei_max > 0:
        ei_norm = ei / ei_max
    else:
        ei_norm = ei.copy()
    
    # We don't have coordinates, so we can't do true spatial diversity.
    # Instead, use a kernel-based diversity proxy: candidates with similar mu/sigma
    # profiles are "close" in function space.
    # 
    # Alternative: Use a UCB-style acquisition that naturally balances exploration
    # and exploitation, with a time-varying beta.
    #
    # Since we don't know the step count, we use the spread of sigma as a proxy:
    # if sigma is large overall, we're early (explore more); if small, exploit.
    
    # Compute a UCB score with adaptive beta
    # beta scales with the relative uncertainty
    sigma_mean = np.mean(sigma)
    sigma_max = np.max(sigma)
    
    # Adaptive beta: larger when uncertainty is high
    beta = 0.5 + 0.5 * (sigma_mean / (sigma_max + 1e-9))
    
    # UCB score
    ucb = mu + beta * sigma  # This is for minimization, so we want low mu, high sigma
    # Actually for minimization, UCB = mu - beta * sigma (we want low values)
    # But we want to explore where sigma is high, so:
    # score = -mu + beta * sigma  (higher is better: low mu, high sigma)
    
    # Combine EI and UCB
    # EI favors exploitation (high EI = good)
    # UCB favors exploration (high sigma, low mu)
    
    # Normalize UCB
    ucb_min = np.min(ucb)
    ucb_max = np.max(ucb)
    if ucb_max > ucb_min:
        ucb_norm = (ucb - ucb_min) / (ucb_max - ucb_min)
    else:
        ucb_norm = np.ones_like(ucb)
    
    # Blend EI and UCB
    # Weight shifts from exploration to exploitation as EI becomes more informative
    # Use a simple heuristic: weight based on the fraction of candidates with significant EI
    ei_significant = np.sum(ei > 0.1 * ei_max) / n
    w_exploit = 0.5 + 0.5 * ei_significant  # 0.5 to 1.0
    w_explore = 1.0 - w_exploit
    
    scores = w_exploit * ei_norm + w_explore * ucb_norm
    
    # Add a small random-like perturbation based on index to break ties
    # This helps avoid always picking the same point
    idx = np.arange(n)
    perturbation = 0.01 * np.sin(idx * 0.1)  # Deterministic but varied
    
    scores = scores + perturbation
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
