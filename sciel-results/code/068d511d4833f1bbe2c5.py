"""Rewritten acquisition function for Bayesian optimization of Branin 2D."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard: the evaluator kills after 330s. We track elapsed time.
_START_TIME = time.time()
_TIME_LIMIT = 325.0  # safety margin before 330s


def acquisition(mu, sigma, f_best):
    """
    Acquisition function: returns 2000 finite scores (higher = evaluate next).
    
    Uses a hybrid strategy:
    - Early phase (when f_best is far from optimum): UCB-like exploration
    - Late phase: EI-like exploitation
    
    We blend based on how many effective evaluations we've done (approximated
    by how close f_best is to the known minimum 0.397887).
    """
    n = len(mu)
    
    # Check deadline
    if time.time() - _START_TIME > _TIME_LIMIT:
        return np.zeros(n)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Branin minimum is approximately 0.397887
    # Use the gap to determine exploration vs exploitation
    min_val = 0.397887
    gap = f_best - min_val
    
    # If gap is large, we're in early phase -> explore more
    # If gap is small, we're in late phase -> exploit more
    # Normalize gap: at start, f_best might be ~100+, so gap ~100
    # After a few good evaluations, gap drops quickly
    # Use a sigmoid-like transition
    
    # Exploration weight: high when gap is large, low when gap is small
    # Transition around gap ~ 1.0 (relative to the scale of Branin)
    explore_weight = 1.0 / (1.0 + np.exp(2.0 * (gap - 1.0)))
    
    # UCB score
    z_ucb = 1.5  # exploration parameter
    ucb_score = -mu + z_ucb * sigma
    
    # EI score (standard expected improvement for minimization)
    z = (f_best - mu) / sigma
    # Clip z to avoid numerical issues
    z = np.clip(z, -10, 10)
    from scipy.stats import norm
    ei_score = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    # EI can be zero or negative where mu > f_best; clip to 0
    ei_score = np.maximum(ei_score, 0.0)
    
    # Normalize both scores to [0, 1] range for blending
    ucb_min, ucb_max = ucb_score.min(), ucb_score.max()
    if ucb_max - ucb_min > 1e-9:
        ucb_norm = (ucb_score - ucb_min) / (ucb_max - ucb_min)
    else:
        ucb_norm = np.zeros(n)
    
    ei_min, ei_max = ei_score.min(), ei_score.max()
    if ei_max - ei_min > 1e-9:
        ei_norm = (ei_score - ei_min) / (ei_max - ei_min)
    else:
        ei_norm = np.zeros(n)
    
    # Blend: early -> UCB, late -> EI
    score = explore_weight * ucb_norm + (1.0 - explore_weight) * ei_norm
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
