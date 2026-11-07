"""
Program using a Pareto-frontier selection criterion for Bayesian optimization.
This approach computes Expected Improvement (EI) and Upper Confidence Bound (UCB)
independently, then selects candidates based on the minimum of their normalized
ranks. This favors points that are both highly promising (high EI) and highly
uncertain (high UCB), avoiding the linear blending pitfalls of standard hybrids.
"""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = 330.0
_SAFETY_MARGIN = 5.0

def acquisition(mu, sigma, f_best):
    """
    Pareto-frontier selection acquisition function.
    
    Parameters:
    -----------
    mu : array-like
        Mean of the GP posterior at candidate points.
    sigma : array-like
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best observed value so far.
        
    Returns:
    --------
    scores : ndarray
        Scores for each candidate, higher meaning evaluate next.
    """
    # Deadline guard
    if time.time() > _DEADLINE - _SAFETY_MARGIN:
        # Return a safe fallback: just minimize mu (greedy)
        return -mu
    
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    
    # 1. Compute Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    z = (f_best - mu) / sigma_safe
    ei_scores = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # 2. Compute Upper Confidence Bound (UCB)
    # For minimization, we want low mu and high sigma.
    # UCB = mu - kappa * sigma
    # We want to select points with LOW UCB values (high score in maximization context).
    # Let's define ucb_scores such that higher is better for the "promising/uncertain" metric.
    # Actually, standard UCB for minimization is mu - kappa*sigma.
    # Points with low mu - kappa*sigma are good.
    # To fit the "maximize score" framework of the rank selection,
    # let's define the "UCB potential" as - (mu - kappa * sigma) = kappa * sigma - mu.
    # Or simply use the raw UCB value and invert the rank logic later.
    # Let's stick to: ucb_value = mu - kappa * sigma.
    # We want to select candidates with LOW ucb_value.
    kappa = 2.0
    ucb_values = mu - kappa * sigma_safe
    
    # 3. Compute Ranks
    # We want to find candidates that are:
    # a) High EI (High rank in ei_scores)
    # b) Low UCB (Low rank in ucb_values, which corresponds to High rank in -ucb_values)
    
    # Normalize ranks to [0, 1]
    # Rank 0 is worst, Rank N-1 is best.
    # For EI: Higher is better.
    ei_ranks = np.argsort(np.argsort(ei_scores)).astype(float)
    ei_ranks_norm = ei_ranks / (len(ei_ranks) - 1) if len(ei_ranks) > 1 else np.ones_like(ei_ranks)
    
    # For UCB: Lower is better.
    # So we invert the rank: best (lowest ucb_value) should have high rank.
    ucb_ranks_raw = np.argsort(np.argsort(ucb_values)).astype(float)
    ucb_ranks_norm = 1.0 - (ucb_ranks_raw / (len(ucb_ranks_raw) - 1) if len(ucb_ranks_raw) > 1 else np.zeros_like(ucb_ranks_raw))
    
    # 4. Pareto Frontier Selection Criterion
    # Score = min(ei_ranks_norm, ucb_ranks_norm)
    # This selects points that are simultaneously in the top tier of both metrics.
    # If a point has high EI but low UCB rank (not very uncertain), min is low.
    # If a point has low EI but high UCB rank, min is low.
    # Only points with BOTH high EI rank and high UCB rank get high scores.
    
    scores = np.minimum(ei_ranks_norm, ucb_ranks_norm)
    
    # Ensure finite
    if not np.all(np.isfinite(scores)):
        finite_mask = np.isfinite(scores)
        if np.any(finite_mask):
            mean_val = np.mean(scores[finite_mask])
            scores = np.where(finite_mask, scores, mean_val)
        else:
            scores = np.zeros_like(mu)
            
    return scores
# EVOLVE-BLOCK-END
