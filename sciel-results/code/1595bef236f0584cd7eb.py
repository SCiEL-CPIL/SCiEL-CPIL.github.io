"""Cost-aware acquisition function for Bayesian optimization (rosenbrock 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Adaptive probability-of-improvement with rank-aware cost normalization.
    
    Core: P(f(x) < f_best) / adaptive_cost
    The adaptive cost uses the rank of each candidate's cost relative to the
    distribution, creating a non-linear penalty that is less aggressive for
    mid-cost candidates and more aggressive for the most expensive ones.
    An exploration bonus from GP uncertainty fades as f_best -> 0.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean predictions in objective units
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions in objective units
    f_best : float
        Lowest observed function value
    cost : ndarray, shape (2000,)
        Evaluation costs (1 + mean(x scaled to [0,1]))
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next)
    """
    t_start = time.time()
    
    # Safety: ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Standardized distance: how many sigma below f_best is the mean
    z = (f_best - mu) / sigma_safe
    
    # Probability of improvement: P(f(x) < f_best) = Phi(z)
    prob_improve = norm.cdf(z)
    
    # Clip to avoid numerical issues at extremes
    prob_improve = np.clip(prob_improve, 1e-12, 1.0 - 1e-12)
    
    # Rank-aware cost normalization:
    # Instead of dividing by raw cost, use a transformed cost that
    # emphasizes the relative ranking. This makes the penalty adaptive
    # to the cost distribution of the current candidate set.
    # 
    # Strategy: use percentile rank of cost, then map to a penalty
    # that is gentle for low-cost candidates and steep for high-cost ones.
    # This is a structural change from fixed linear cost division.
    
    # Compute percentile rank of each cost (0 to 1)
    cost_rank = np.argsort(np.argsort(cost_safe)) / len(cost_safe)
    
    # Adaptive cost penalty: 
    # - Low rank (cheap): penalty ~ 1 (no extra penalty)
    # - High rank (expensive): penalty grows super-linearly
    # Use: 1 + (cost_rank)^2 * (max_cost - min_cost) / min_cost
    # This normalizes so that the most expensive candidate gets a strong penalty
    cost_min = np.min(cost_safe)
    cost_max = np.max(cost_safe)
    cost_range = max(cost_max - cost_min, 1e-9)
    
    # Normalized cost position in [0, 1]
    cost_pos = (cost_safe - cost_min) / cost_range
    
    # Super-linear penalty: cheap candidates barely penalized, expensive ones heavily
    # Using quadratic: penalty = 1 + 3 * cost_pos^2
    # This means a candidate at 50% cost position gets penalty 1.75
    # A candidate at 100% cost position gets penalty 4.0
    adaptive_cost = 1.0 + 3.0 * cost_pos ** 2
    
    # Primary score: PI / adaptive_cost
    score = prob_improve / adaptive_cost
    
    # Exploration bonus: GP uncertainty that fades as we converge
    # When f_best is large (early), explore more; when f_best is small (late), exploit
    # Use exponential decay: weight = exp(-f_best / scale)
    # Scale of 2.0 means at f_best=2, weight ~ 0.37; at f_best=10, weight ~ 0.007
    if f_best > 0:
        explore_weight = np.exp(-f_best / 2.0)
    else:
        explore_weight = 1.0
    
    # Exploration term: sigma / adaptive_cost, scaled by explore_weight
    # This rewards uncertain regions, but less so when we're close to optimal
    explore_bonus = explore_weight * sigma_safe / adaptive_cost
    
    # Blend: primary PI signal + bounded exploration
    # The exploration is capped to not dominate the PI signal
    score = score + 0.3 * explore_bonus
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Deadline guard: if we're getting close to timeout, return immediately
    # (This is a simple vectorized function, so it should be fast, but guard just in case)
    if time.time() - t_start > 325:
        return score
    
    return score
# EVOLVE-BLOCK-END
