"""Acquisition function for Bayesian optimization of ackley 2d."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_start_time = time.time()
_DEADLINE_MARGIN = 10.0
_TOTAL_BUDGET = 330.0


def _time_remaining():
    return _TOTAL_BUDGET - _DEADLINE_MARGIN - (time.time() - _start_time)


def acquisition(mu, sigma, f_best):
    """
    Probability of Improvement with adaptive exploration bonus.
    
    Uses a deterministic PI score combined with an uncertainty-based exploration
    bonus that adapts to the current best value. This is a structural change from
    the previous TS+EI hybrid to a pure PI-based approach with explicit
    exploration-exploitation tradeoff.
    
    Parameters:
        mu: array of shape (2000,) - posterior mean
        sigma: array of shape (2000,) - posterior std
        f_best: float - best observed value so far
    
    Returns:
        scores: array of shape (2000,) - higher means evaluate next
    """
    # Deadline guard: if time is running out, return pure exploitation
    if _time_remaining() < 5.0:
        return -mu.copy()
    
    n = len(mu)
    
    # Safe sigma to avoid division by zero
    safe_sigma = np.where(sigma < 1e-8, 1e-8, sigma)
    
    # Probability of Improvement for minimization: P(f(x) < f_best)
    # = P(N(mu, sigma) < f_best) = Phi((f_best - mu) / sigma)
    z_pi = (f_best - mu) / safe_sigma
    pi = norm.cdf(z_pi)
    pi = np.nan_to_num(pi, nan=0.5, posinf=1.0, neginf=0.0)
    
    # Exploration bonus: scaled by uncertainty and adaptive factor
    # The exploration bonus encourages exploring high-uncertainty regions
    # The adaptive factor decreases as we get closer to the optimum
    # When f_best is large (early), we explore more; when small (late), we exploit
    progress = f_best / (f_best + 1.0)
    # Exploration weight: high early, low late
    expl_weight = 0.5 * progress + 0.1
    
    # Scale the exploration bonus by the typical improvement scale
    # Use the median sigma as a reference scale
    med_sigma = np.median(sigma)
    if med_sigma < 1e-10:
        med_sigma = 1e-10
    
    # Exploration bonus: normalized uncertainty
    expl_bonus = sigma / med_sigma
    
    # Combine: PI (exploitation) + weighted exploration bonus
    # Scale PI to [0, 1] range and combine with exploration
    scores = pi + expl_weight * expl_bonus
    
    # Normalize to comparable scale using min-max
    min_s = np.min(scores)
    max_s = np.max(scores)
    range_s = max_s - min_s
    if range_s < 1e-10:
        scores = scores - min_s
    else:
        scores = (scores - min_s) / range_s
    
    # Ensure finite values
    scores = np.nan_to_num(scores, nan=0.5, posinf=1.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
