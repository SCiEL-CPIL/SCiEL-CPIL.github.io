import numpy as np
from scipy.stats import norm
import time

_start_time = None
_call_count = 0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization on Shekel 4D.
    
    Args:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP standard deviation predictions
        f_best: float - best observed function value so far
        cost: array of shape (2000,) - evaluation costs for each candidate
    
    Returns:
        array of shape (2000,) - acquisition scores (higher = better)
    """
    global _start_time, _call_count
    
    if _start_time is None:
        _start_time = time.time()
    _call_count += 1
    
    elapsed = time.time() - _start_time
    if elapsed > 325:
        # Safety: return exploit-heavy fallback
        cost_safe = np.maximum(cost, 1e-9)
        return -mu / (cost_safe + 1e-9)
    
    # Clip sigma to avoid numerical issues
    sigma = np.clip(sigma, 1e-9, 1e6)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Compute z-score for EI
    z = (f_best - mu) / sigma
    
    # Expected Improvement (EI) for minimization
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware EI (maximize)
    objective1 = ei / cost_safe
    
    # Cost to minimize
    objective2 = cost_safe
    
    # Pareto front-based selection:
    # A candidate is dominated if another has >= EI/cost AND <= cost
    # We compute dominance counts (number of candidates that dominate each candidate)
    # Lower dominance count = better (Pareto-optimal candidates have 0)
    
    n = len(objective1)
    dominance_counts = np.zeros(n, dtype=np.int32)
    
    # Vectorized dominance computation in chunks to save memory
    # For each candidate i, count how many j satisfy:
    #   objective1[j] >= objective1[i] AND objective2[j] <= objective2[i]
    #   AND (objective1[j] > objective1[i] OR objective2[j] < objective2[i])
    
    chunk_size = 500
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        o1_chunk = objective1[start:end]  # (chunk,)
        o2_chunk = objective2[start:end]  # (chunk,)
        
        # o1_all >= o1_chunk (broadcast: (chunk, n))
        cond1 = (objective1[None, :] >= o1_chunk[:, None])
        # o2_all <= o2_chunk (broadcast: (chunk, n))
        cond2 = (objective2[None, :] <= o2_chunk[:, None])
        # Strict improvement in at least one objective
        cond3 = ((objective1[None, :] > o1_chunk[:, None]) | 
                 (objective2[None, :] < o2_chunk[:, None]))
        
        dominated_by = cond1 & cond2 & cond3
        dominance_counts[start:end] = dominated_by.sum(axis=1)
    
    # Score: higher for fewer dominators, with tie-breaking by EI/cost
    # Use rank-based scoring: non-dominated get highest scores
    # Invert dominance counts so lower count = higher score
    # Add small EI/cost term to break ties among same-dominance candidates
    
    # Normalize dominance counts to [0, 1] range (inverted)
    max_dom = dominance_counts.max()
    if max_dom > 0:
        dom_score = 1.0 - dominance_counts / (max_dom + 1)
    else:
        dom_score = np.ones(n)
    
    # Normalize objective1 for tie-breaking
    o1_min, o1_max = objective1.min(), objective1.max()
    if o1_max > o1_min:
        o1_norm = (objective1 - o1_min) / (o1_max - o1_min)
    else:
        o1_norm = np.ones(n)
    
    # Final score: dominance-based ranking + EI/cost tie-breaker
    # Weight: 90% dominance rank, 10% normalized EI/cost
    scores = 0.9 * dom_score + 0.1 * o1_norm
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
