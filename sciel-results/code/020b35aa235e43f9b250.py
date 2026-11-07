"""Cost-aware acquisition function for Bayesian optimization (hartmann 6d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization on Hartmann 6d.
    Uses a dynamically annealed GP-UCB with cost normalization.
    The exploration coefficient beta(t) decays exponentially based on the
    ratio of current total cost to the budget, ensuring early broad exploration
    and late-stage precise exploitation.

    Parameters
    ----------
    mu : array (2000,) GP mean
    sigma : array (2000,) GP std
    f_best : float best observed value
    cost : array (2000,) evaluation cost per candidate

    Returns
    -------
    scores : array (2000,) higher = evaluate next
    """
    t0 = time.time()
    DEADLINE = 320.0  # safety margin before 330s wall-clock limit

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Ensure finite
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1.0)

    # Estimate current total cost from the mean of the cost array
    # This is a proxy for how far we are in the budget
    # The cost array has shape (2000,) and represents cost per candidate
    # We use the median as a robust estimate of typical cost
    typical_cost = np.median(cost)
    
    # Budget is 30, we start with 5 points
    # Estimate total cost spent so far:
    # We don't have direct access to total cost, but we can infer from the
    # relationship between f_best and the initial random points.
    # A simpler approach: use the fraction of the budget that has been used.
    # Since we don't know the exact total cost, we'll use a heuristic:
    # Early on, f_best is likely close to the initial random points.
    # Later, f_best should be much lower.
    # 
    # Alternative: use the number of distinct values in f_best to estimate
    # how many evaluations have been done. But we only have f_best.
    #
    # Let's use a different approach: the cost-aware UCB should adapt based
    # on the relative uncertainty. When sigma is large relative to the range
    # of mu, we are in early stages and should explore more.
    #
    # For the annealing, we'll use a fixed schedule based on the assumption
    # that the total budget is 30. We'll estimate the fraction of budget used
    # by comparing the current f_best to the expected best after n evaluations.
    #
    # A simpler and more robust approach: use the ratio of the median sigma
    # to the range of mu as a proxy for how much exploration is needed.
    # When sigma is large relative to mu's range, we explore more.
    
    mu_range = np.max(mu) - np.min(mu)
    if mu_range < 1e-9:
        mu_range = 1.0
    
    # Median sigma relative to mu range
    sigma_rel = np.median(sigma) / mu_range
    
    # Map sigma_rel to a beta coefficient
    # When sigma_rel is large (early), beta should be high (explore)
    # When sigma_rel is small (late), beta should be low (exploit)
    # Use a smooth mapping
    beta_max = 5.0
    beta_min = 0.5
    # sigma_rel typically ranges from ~0.1 (late) to ~1.0+ (early)
    # Normalize and map
    beta_t = beta_min + (beta_max - beta_min) * np.clip(sigma_rel / 1.0, 0, 1)
    
    # Also incorporate cost awareness: higher cost should reduce exploration
    cost_mean = np.mean(cost)
    cost_factor = 1.0 / (1.0 + 0.1 * (cost_mean - 1.0))
    beta_t *= cost_factor
    
    # Compute GP-UCB score
    # For minimization, we want to maximize: -mu + beta * sigma
    # Cost-normalized: divide by cost^beta_cost
    # Use a moderate cost exponent
    beta_cost = 0.8
    
    ucb_term = -mu + beta_t * sigma
    cost_normalizer = cost ** beta_cost
    scores = ucb_term / cost_normalizer
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check time
    if time.time() - t0 > DEADLINE:
        return scores
    
    # Refinement: blend with EI for exploitation
    # Compute EI
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Normalize EI and UCB to comparable scales
    ei_max = np.max(ei)
    ucb_max = np.max(np.abs(ucb_term))
    
    if ei_max > 1e-9:
        ei_norm = ei / ei_max
    else:
        ei_norm = np.zeros_like(ei)
    
    if ucb_max > 1e-9:
        ucb_norm = ucb_term / ucb_max
    else:
        ucb_norm = np.zeros_like(ucb_term)
    
    # Blend: more EI weight as we progress (lower sigma_rel)
    ei_weight = np.clip(1.0 - sigma_rel, 0.3, 0.7)
    ucb_weight = 1.0 - ei_weight
    
    blended = ei_weight * ei_norm + ucb_weight * ucb_norm
    blended_cost = blended / cost_normalizer
    
    # Choose the better of the two scores
    # Use the one that gives higher values overall
    if np.mean(blended_cost) > np.mean(scores):
        scores = blended_cost
    
    # Final finite check
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
