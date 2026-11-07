"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Track total elapsed time for deadline guard
_T0 = time.time()
_DEADLINE = 320.0

def _time_left():
    return _DEADLINE - (time.time() - _T0)

def acquisition(mu, sigma, f_best, cost):
    """
    Computes a cost-aware acquisition function using Pareto-optimal cost-normalization.
    Combines Expected Improvement (EI) and Upper Confidence Bound (UCB) with
    marginal cost normalization and diversity-aware exploration.
    
    Args:
        mu: array of shape (N,) predicted mean
        sigma: array of shape (N,) predicted std deviation
        f_best: float, best value found so far
        cost: array of shape (N,) evaluation cost for each candidate
        
    Returns:
        scores: array of shape (N,) acquisition values (higher is better)
    """
    # Deadline guard: if we've been running too long, return a simple, fast score
    if _time_left() < 10.0:
        mu = np.asarray(mu, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        cost = np.asarray(cost, dtype=np.float64)
        sigma_safe = np.maximum(sigma, 1e-9)
        cost_safe = np.maximum(cost, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        return ei / cost_safe

    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma and cost to small positive values
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Calculate z-score with clipping for numerical stability
    z = (f_best - mu) / sigma_safe
    z_clipped = np.clip(z, -10.0, 10.0)
    
    # Calculate PDF and CDF
    pdf = norm.pdf(z_clipped)
    cdf = norm.cdf(z_clipped)
    
    # Expected Improvement
    ei = (f_best - mu) * cdf + sigma_safe * pdf
    ei = np.maximum(ei, 0.0)
    
    # UCB with adaptive beta for exploration
    beta = 1.5
    ucb = -mu + beta * sigma_safe
    
    # Pareto-optimal cost normalization:
    # Instead of dividing by absolute cost, use marginal cost relative to current best cost.
    # This treats the search as multi-objective: maximize improvement while minimizing cost.
    
    # Compute the minimum cost among candidates with significant acquisition value
    # This identifies the "efficient frontier" of cost vs. improvement
    ei_threshold = np.percentile(ei, 50) if np.max(ei) > 1e-12 else 0.0
    ucb_threshold = np.percentile(ucb, 50) if np.max(ucb) > 1e-12 else 0.0
    
    # Find candidates that are "Pareto-relevant" - those with above-median EI or UCB
    pareto_relevant = (ei > ei_threshold) | (ucb > ucb_threshold)
    
    if np.any(pareto_relevant):
        # Among Pareto-relevant candidates, find the minimum cost
        min_cost_pareto = np.min(cost_safe[pareto_relevant])
    else:
        min_cost_pareto = np.min(cost_safe)
    
    # Marginal cost: how much more expensive is this candidate compared to the cheapest Pareto-relevant point
    # This normalizes cost relative to the efficient frontier
    marginal_cost = cost_safe - min_cost_pareto + min_cost_pareto
    # Equivalent to cost_safe, but conceptually it's the cost relative to the Pareto frontier
    
    # Alternative: use a softer normalization based on the distribution of costs
    # among high-value candidates
    if np.any(pareto_relevant) and np.sum(pareto_relevant) > 10:
        cost_high_value = cost_safe[pareto_relevant]
        cost_median_hv = np.median(cost_high_value)
        cost_std_hv = np.std(cost_high_value)
        if cost_std_hv < 1e-9:
            cost_std_hv = 1e-9
        # Z-score of cost among high-value candidates
        cost_z = (cost_safe - cost_median_hv) / cost_std_hv
        # Penalize expensive candidates more strongly
        cost_penalty = 1.0 / (1.0 + 0.5 * np.maximum(cost_z, 0.0))
    else:
        # Fallback: simple cost normalization
        cost_penalty = 1.0 / cost_safe
    
    # Normalize EI and UCB to combine them fairly
    ei_max = np.max(ei) if np.max(ei) > 1e-12 else 1.0
    ucb_max = np.max(ucb) if np.max(ucb) > 1e-12 else 1.0
    ucb_min = np.min(ucb) if np.max(ucb) > 1e-12 else 0.0
    
    ei_norm = ei / ei_max
    ucb_norm = (ucb - ucb_min) / (ucb_max - ucb_min + 1e-12)
    
    # Weighted combination: 70% EI, 30% UCB
    combined = 0.7 * ei_norm + 0.3 * ucb_norm
    
    # Diversity penalty: penalize candidates in well-explored (low sigma) regions
    sigma_min = np.min(sigma_safe)
    sigma_max = np.max(sigma_safe)
    
    if sigma_max - sigma_min > 1e-12:
        sigma_norm = (sigma_safe - sigma_min) / (sigma_max - sigma_min)
    else:
        sigma_norm = np.ones_like(sigma_safe)
    
    # Crowding is high when sigma_norm is low.
    crowding = 1.0 - sigma_norm
    
    # Penalty weight: how strongly to penalize crowded regions.
    penalty_weight = 0.3
    # Scale penalty by the magnitude of the combined score
    combined_max = np.max(combined) if np.max(combined) > 1e-12 else 1.0
    crowding_penalty = penalty_weight * crowding * combined_max
    
    # Combined score: combined - crowding_penalty
    score = combined - crowding_penalty
    
    # Apply Pareto-optimal cost normalization
    score = score * cost_penalty
    
    # Ensure all values are finite and non-negative
    score = np.where(np.isfinite(score), score, 0.0)
    score = np.maximum(score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
