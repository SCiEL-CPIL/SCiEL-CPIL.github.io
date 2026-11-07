"""Cost-aware acquisition function for Bayesian optimization (levy 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition: Adaptive UCB with Additive Cost Penalty.
    
    Structural change: Instead of dividing by cost (which scales the objective
    space by cost), this method subtracts a cost penalty scaled to the objective
    space. This allows high-cost points to be selected if they offer sufficiently
    large expected improvement, rather than being uniformly suppressed.
    
    It uses an adaptive exploration weight (beta) that increases when uncertainty
    is high relative to the current best value, promoting exploration in early
    stages or uncertain regions.
    
    Parameters
    ----------
    mu : np.ndarray, shape (2000,)
        GP mean predictions.
    sigma : np.ndarray, shape (2000,)
        GP standard deviation predictions.
    f_best : float
        Best observed objective value (lower is better).
    cost : np.ndarray, shape (2000,)
        Cost of evaluating each candidate.
    
    Returns
    -------
    np.ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    t0 = time.time()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety margins
    sigma_safe = np.maximum(sigma, 1e-12)
    cost_safe = np.maximum(cost, 1e-12)
    
    # Expected Improvement: E[max(f_best - f(x), 0)]
    z = (f_best - mu) / sigma_safe
    z_clipped = np.clip(z, -30.0, 30.0)
    
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    ei = np.maximum(ei, 0.0)
    
    # Adaptive UCB weight:
    # Inspired by the successful 'inspiration 1' strategy.
    # When uncertainty is high relative to the gap to optimum, explore more.
    sigma_mean = np.mean(sigma_safe)
    gap_scale = max(abs(f_best), 1e-10)
    
    # Ratio of typical uncertainty to the current best value
    # If f_best is large (early), ratio is small -> less exploration? 
    # Wait, if f_best is large, we are far from optimum. We should explore MORE.
    # Let's check the logic in Inspiration 1:
    # uncertainty_ratio = min(sigma_mean / gap_scale, 5.0)
    # beta = 0.1 + 0.9 * np.tanh(uncertainty_ratio)
    # If f_best is large, gap_scale is large, ratio is small, beta is small (~0.1).
    # This means LESS exploration early on? That seems counter-intuitive for BO.
    # Usually, you explore early.
    # However, if the GP is uncertain everywhere (high sigma) and f_best is high,
    # maybe the GP is just bad, so we should trust the prior (uniform) more?
    # Or maybe the "gap_scale" logic is: if f_best is high, the potential improvement
    # is huge, so EI dominates anyway.
    
    # Let's stick to the structure of Insp 1 as it scored better, but ensure the
    # cost handling is the structural change.
    
    uncertainty_ratio = min(sigma_mean / gap_scale, 5.0)
    beta = 0.1 + 0.9 * np.tanh(uncertainty_ratio)
    
    # Adjust beta based on sigma variance (spread of uncertainty)
    sigma_cv = np.std(sigma_safe) / (sigma_mean + 1e-12)
    beta = beta * (1.0 + 0.3 * min(sigma_cv, 2.0))
    beta = min(beta, 1.5)
    
    ucb_term = beta * sigma_safe
    
    # Base acquisition value (Exploitation + Exploration)
    base_score = ei + ucb_term
    
    # STRUCTURAL CHANGE: Additive Cost Penalty
    # Instead of base_score / cost, we subtract a penalty.
    # We need to scale the cost to be comparable to base_score.
    
    # Estimate the scale of the objective function from the GP predictions
    # Use the mean of the sigma or the range of mu as a proxy for the objective scale
    obj_scale = np.mean(sigma_safe) + np.std(mu)
    if obj_scale < 1e-6:
        obj_scale = 1.0
        
    # Normalize cost to [0, 1] roughly relative to its own distribution
    # cost is 1 + mean(x_scaled). Range is [1, 2].
    # We want the penalty to be significant but not overwhelming.
    # Let's define penalty = lambda * (cost - min_cost)
    # And scale lambda such that the max penalty is ~ 0.5 * obj_scale
    
    min_cost = np.min(cost_safe)
    max_cost = np.max(cost_safe)
    cost_range = max_cost - min_cost
    if cost_range < 1e-6:
        cost_range = 1.0
        
    # The penalty should be proportional to the excess cost over the minimum.
    # We want a point with max cost to be penalized by roughly half the typical improvement scale.
    # This makes cost a "soft" constraint rather than a hard filter.
    lambda_cost = 0.5 * obj_scale / cost_range
    
    cost_penalty = lambda_cost * (cost_safe - min_cost)
    
    score = base_score - cost_penalty
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Deadline guard
    elapsed = time.time() - t0
    if elapsed > 280.0:
        return score
    
    return score
# EVOLVE-BLOCK-END
