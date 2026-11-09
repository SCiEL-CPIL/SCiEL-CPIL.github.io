"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = 300.0  # 30s safety margin before 330s kill

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Pareto filtering and Log-Cost weighted EI.
    
    Parameters
    ----------
    mu : np.ndarray, shape (2000,)
        GP mean predictions
    sigma : np.ndarray, shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best observed value so far
    cost : np.ndarray, shape (2000,)
        Cost of evaluating each candidate
    
    Returns
    -------
    np.ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next)
    """
    # Deadline guard
    if time.time() - _start_time > _DEADLINE:
        return np.zeros(len(mu))
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure finite inputs
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Compute Expected Improvement for minimization
    improvement = f_best - mu
    z = improvement / sigma
    ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    n = len(mu)
    
    # Structural Change: Use Log-Cost penalty instead of linear division
    # This changes the trade-off between improvement and cost.
    # High cost is penalized less severely than in EI/cost.
    log_cost = np.log1p(cost)  # log(1 + cost)
    
    # Score = EI / log(1 + cost)
    scores = ei / log_cost
    
    # Pareto filtering based on (EI, Log-Cost)
    # A point is dominated if there is another point with >= EI and <= Log-Cost
    # Sort by EI descending
    ei_sorted_idx = np.argsort(-ei)
    
    # Initialize scores to 0 for non-front points
    final_scores = np.zeros(n)
    
    min_log_cost_so_far = np.inf
    front_mask = np.zeros(n, dtype=bool)
    
    for idx in ei_sorted_idx:
        lc = log_cost[idx]
        if lc < min_log_cost_so_far:
            front_mask[idx] = True
            min_log_cost_so_far = lc
    
    # Only keep scores for Pareto-optimal points
    final_scores[front_mask] = scores[front_mask]
    
    # Ensure finite output
    final_scores = np.where(np.isfinite(final_scores), final_scores, 0.0)
    
    return final_scores
# EVOLVE-BLOCK-END
