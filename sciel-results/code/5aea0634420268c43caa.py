"""Cost-aware acquisition function for Bayesian optimization (powell 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE_MARGIN = 3.0  # seconds before the hard 330s limit


def _elapsed():
    return time.time() - _START_TIME


def _deadline_exceeded():
    return _elapsed() > (330.0 - _DEADLINE_MARGIN)


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining Expected Improvement with a
    probability-of-improvement term weighted by uncertainty.
    
    Strategy: Use a blend of EI/cost and POI/cost, where POI adds exploration
    pressure. The POI term is scaled by sigma to balance exploitation (EI)
    and exploration (POI weighted by uncertainty).
    
    We use: score = (EI + beta * sigma * POI) / cost
    where beta controls the exploration-exploitation tradeoff.
    
    A small constant offset in the numerator prevents division by zero and
    ensures finite outputs even when cost is very small.
    """
    # Explicit deadline guard
    if _deadline_exceeded():
        # Return a safe, finite fallback using simple EI
        mu = np.asarray(mu, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        cost = np.asarray(cost, dtype=np.float64)
        sigma_safe = np.maximum(sigma, 1e-9)
        improvement = f_best - mu
        z = improvement / sigma_safe
        ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
        ei = np.where(np.isfinite(ei), ei, 0.0)
        return ei / (cost + 1e-9)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Guard against non-finite or invalid values
    mu = np.where(np.isfinite(mu), mu, 1e10)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)

    # Guard against zero/near-zero sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standardized improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # Probability of Improvement
    poi = norm.cdf(z)
    poi = np.where(np.isfinite(poi), poi, 0.0)
    
    # Adaptive beta: reduce exploration pressure as we get closer to the optimum
    # When f_best is large (far from optimum), use more exploration (higher beta)
    # When f_best is small (close to optimum), use less exploration (lower beta)
    # Use a smooth sigmoid transition centered around f_best = 0.5
    t_exploit = 1.0 / (1.0 + np.exp(-4.0 * (f_best - 0.5)))
    # beta ranges from 0.8 (early, high f_best) to 0.2 (late, low f_best)
    beta = 0.2 + 0.6 * (1.0 - t_exploit)
    
    # Blend: EI dominates when confident, POI*sigma adds exploration
    numerator = ei + beta * sigma_safe * poi
    
    # Cost normalization with a softer penalty using sqrt(cost) instead of cost
    # This provides a gentler penalty for expensive evaluations while still
    # preferring cheaper ones
    cost_safe = np.maximum(cost, 1e-6)
    cost_penalty = np.sqrt(cost_safe)
    
    score = numerator / cost_penalty
    
    # Ensure finiteness
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
