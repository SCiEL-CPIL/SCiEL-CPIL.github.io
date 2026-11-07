"""Cost-aware acquisition function for Bayesian optimization (ackley 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard: track start time at module load
_T0 = time.time()
_DEADLINE = 295.0  # seconds, safety margin before 300s limit


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition score for 2000 candidates.
    
    Combines Expected Improvement (EI) with a UCB exploration bonus,
    both normalized by cost. This balances exploitation (EI) with
    exploration (UCB) while penalizing expensive evaluations.
    
    Parameters
    ----------
    mu : ndarray, shape (2000,)
        GP mean predictions in objective units.
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions in objective units.
    f_best : float
        Best observed value so far.
    cost : ndarray, shape (2000,)
        Evaluation costs (in [1, 2]).
    
    Returns
    -------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    # Deadline guard: if we are running out of time, return a safe fallback
    if time.time() - _T0 > _DEADLINE:
        return np.zeros_like(mu)
    
    # Ensure sigma is non-negative and safe for operations
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Ensure cost is positive
    cost_safe = np.maximum(cost, 1e-9)
    
    # Expected Improvement for minimization:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # UCB exploration bonus for minimization:
    # We want to explore where uncertainty is high.
    # UCB bonus = beta * sigma (positive contribution to score)
    # Beta controls exploration strength.
    # With a budget of ~30 evaluations, moderate exploration is needed.
    beta = 0.5
    
    ucb_bonus = beta * sigma_safe
    
    # Combine EI and UCB bonus:
    # Total score before cost normalization
    raw_score = ei + ucb_bonus
    
    # Normalize by cost to prefer cheaper evaluations
    scores = raw_score / cost_safe
    
    # Ensure output is finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
