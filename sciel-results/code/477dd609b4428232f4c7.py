"""Cost-aware acquisition function for Bayesian optimization (shekel 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Pareto front-based selection.
    Computes Expected Improvement (EI) and cost, finds the non-dominated set,
    and scores candidates based on their rank within this front to prioritize
    the best trade-off between improvement and budget efficiency.
    
    Parameters:
    mu: array of shape (2000,)
    sigma: array of shape (2000,)
    f_best: float
    cost: array of shape (2000,)
    
    Returns:
    score: array of shape (2000,)
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = len(mu)
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)
    
    # Compute Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    improvement = f_best - mu
    z = improvement / sigma_safe
    # Clip z to avoid numerical issues
    z = np.clip(z, -10, 10)
    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0)
    
    # We want to maximize EI and minimize cost
    # Normalize EI and cost to [0,1] for fair comparison
    ei_min, ei_max = np.min(ei), np.max(ei)
    cost_min, cost_max = np.min(cost_safe), np.max(cost_safe)
    
    if ei_max > ei_min:
        ei_norm = (ei - ei_min) / (ei_max - ei_min)
    else:
        ei_norm = np.zeros_like(ei)
    
    if cost_max > cost_min:
        cost_norm = (cost_safe - cost_min) / (cost_max - cost_min)
    else:
        cost_norm = np.zeros_like(cost_safe)
    
    # Find non-dominated set: candidate i dominates j if ei_i >= ei_j and cost_i <= cost_j
    # and at least one is strict
    # A candidate is non-dominated if no other candidate has both higher EI and lower cost
    
    # Efficient check: sort by EI descending, then find those with cost less than all previous
    order = np.argsort(-ei_norm)  # sort by EI descending
    ei_sorted = ei_norm[order]
    cost_sorted = cost_norm[order]
    
    # Track minimum cost seen so far
    min_cost_so_far = np.inf
    is_non_dominated = np.zeros(n, dtype=bool)
    
    for idx in range(n):
        i = order[idx]
        if cost_sorted[idx] < min_cost_so_far:
            is_non_dominated[i] = True
            min_cost_so_far = cost_sorted[idx]
    
    # Score: non-dominated points get high scores, others get lower
    # Within non-dominated, prefer those with higher EI (since they're on the front)
    score = np.zeros(n)
    
    if np.any(is_non_dominated):
        # Assign scores to non-dominated points based on their EI
        nd_indices = np.where(is_non_dominated)[0]
        nd_ei = ei_norm[nd_indices]
        
        # Normalize EI among non-dominated to [0,1]
        nd_ei_min, nd_ei_max = np.min(nd_ei), np.max(nd_ei)
        if nd_ei_max > nd_ei_min:
            nd_ei_score = (nd_ei - nd_ei_min) / (nd_ei_max - nd_ei_min)
        else:
            nd_ei_score = np.ones_like(nd_ei)
        
        # Give non-dominated points scores in [0.5, 1.0]
        score[nd_indices] = 0.5 + 0.5 * nd_ei_score
        
        # Give dominated points scores based on how close they are to the front
        # Use a simple heuristic: score proportional to (EI - cost) normalized
        dom_indices = np.where(~is_non_dominated)[0]
        if len(dom_indices) > 0:
            dom_raw = ei_norm[dom_indices] - cost_norm[dom_indices]
            dom_raw_min, dom_raw_max = np.min(dom_raw), np.max(dom_raw)
            if dom_raw_max > dom_raw_min:
                dom_score = (dom_raw - dom_raw_min) / (dom_raw_max - dom_raw_min)
            else:
                dom_score = np.ones_like(dom_raw) * 0.5
            # Scale dominated scores to [0, 0.4]
            score[dom_indices] = 0.4 * dom_score
    
    # Sanitize
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
