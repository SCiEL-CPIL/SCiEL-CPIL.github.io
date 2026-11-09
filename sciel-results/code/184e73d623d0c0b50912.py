# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Module-level deadline tracking
_start_time = None

def _get_elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean of the GP posterior
    sigma : ndarray, shape (2000,)
        Standard deviation of the GP posterior
    f_best : float
        Best observed function value
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate point
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next)
    """
    # Deadline guard: if we're close to timeout, return a safe result
    if _get_elapsed() > 320.0:
        # Return simple EI/cost as fallback
        sigma_safe = sigma + 1e-10
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        scores = ei / (cost + 1e-10)
        return np.where(np.isfinite(scores), scores, 0.0)
    
    # Ensure inputs are valid
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = len(mu)
    
    # Safety: handle edge cases
    if n == 0:
        return np.array([], dtype=np.float64)
    
    # Clip sigma to avoid numerical issues
    sigma = np.maximum(sigma, 1e-10)
    
    # --- Core acquisition: cost-aware Expected Improvement with exploration ---
    
    # 1) Expected Improvement (EI)
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # 2) Cost-normalized EI
    cost_safe = np.maximum(cost, 1e-10)
    ei_cost = ei / cost_safe
    
    # 3) Uncertainty bonus: reward points with high sigma relative to cost
    #    This encourages exploration in cheap, uncertain regions
    uncertainty_bonus = sigma / cost_safe
    
    # 4) Penalty for very high cost points (diminishing returns)
    #    Use a soft penalty that grows with cost
    cost_penalty = 1.0 / (1.0 + cost_safe * 0.1)
    
    # 5) Combine: primary signal is cost-normalized EI, 
    #    with a tunable exploration term
    #    Weight exploration more when EI signal is weak (early in search)
    ei_max = np.max(ei_cost) if np.max(ei_cost) > 0 else 1.0
    
    # Dynamic exploration weight: higher when best EI is small
    # (early in optimization, we want to explore more)
    explore_weight = 0.3 / (1.0 + ei_max * 0.5)
    
    scores = ei_cost * cost_penalty + explore_weight * uncertainty_bonus
    
    # 6) Additional: UCB-like term for points where GP is very uncertain
    #    but only for cheap points
    ucb_term = 1.5 * sigma / cost_safe
    ucb_mask = (cost_safe < np.median(cost_safe)) & (sigma > np.median(sigma))
    scores = np.where(ucb_mask, scores + 0.1 * ucb_term, scores)
    
    # 7) Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # 8) Small random tie-breaking to avoid deterministic behavior
    #    (negligible impact but helps with diversity)
    rng = np.random.RandomState(int(_get_elapsed() * 1000) % (2**31))
    scores += rng.uniform(0, 1e-8, size=n)
    
    return scores
# EVOLVE-BLOCK-END
