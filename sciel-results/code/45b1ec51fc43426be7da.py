"""Cost-aware acquisition function for Bayesian optimization (ackley 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Pareto-frontier filtered acquisition with hypervolume contribution scoring.
    
    mu, sigma, f_best, cost: arrays of shape (2000,)
    Returns: array of 2000 finite scores (higher = evaluate next)
    """
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Guard against non-finite values
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    sigma = np.maximum(sigma, 1e-9)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    cost = np.maximum(cost, 1e-9)
    mu = np.where(np.isfinite(mu), mu, f_best)
    
    # Compute Expected Improvement (EI) for each candidate
    improvement = f_best - mu
    z = improvement / sigma
    # Handle numerical stability
    z = np.clip(z, -10, 10)
    ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.where(np.isfinite(ei), ei, 0.0)
    ei = np.maximum(ei, 0.0)
    
    # Compute "cost savings" proxy: lower cost is better
    # Normalize cost to [0, 1] range for this batch
    cost_min = np.min(cost)
    cost_max = np.max(cost)
    cost_range = cost_max - cost_min
    if cost_range < 1e-9:
        cost_norm = np.zeros_like(cost)
    else:
        cost_norm = (cost - cost_min) / cost_range
    
    # Cost savings: 1 - normalized cost (higher means cheaper)
    cost_savings = 1.0 - cost_norm
    
    # Hypervolume contribution: EI * cost_savings
    # This rewards candidates that offer both high improvement AND low cost
    hv_contribution = ei * cost_savings
    
    # Pareto frontier filter: retain only candidates not dominated in (EI, cost_savings)
    # A candidate is dominated if there exists another with >= EI and >= cost_savings, 
    # with at least one strict inequality
    n = len(ei)
    
    # For efficiency, use a vectorized approach:
    # Candidate i is dominated if exists j: ei[j] >= ei[i] and cs[j] >= cs[i] and (ei[j] > ei[i] or cs[j] > cs[i])
    
    # Sort by EI descending, then by cost_savings descending
    order = np.lexsort((-cost_savings, -ei))
    
    # For each candidate in sorted order, it's on the Pareto front if its cost_savings
    # is greater than all previously seen (which have higher or equal EI)
    max_cs_so_far = -np.inf
    pareto_mask = np.zeros(n, dtype=bool)
    
    for idx in order:
        if cost_savings[idx] > max_cs_so_far:
            pareto_mask[idx] = True
            max_cs_so_far = cost_savings[idx]
    
    # Score: hypervolume contribution for Pareto front candidates, 0 for others
    # Add a small base score for all to keep them finite and non-negative
    base_score = 1e-6
    scores = np.where(pareto_mask, hv_contribution + base_score, base_score)
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, base_score)
    
    return scores
# EVOLVE-BLOCK-END
