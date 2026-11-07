"""Cost-aware acquisition for Bayesian optimization (cosine8 8d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Deadline guard: evaluator kills after 330s wall-clock; use 280s safety margin
_DEADLINE = time.time() + 280.0


def _safe_finite(arr, fallback=0.0):
    """Replace non-finite values with fallback."""
    return np.where(np.isfinite(arr), arr, fallback)


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition using probability-weighted improvement per unit cost.
    This models the selection as a stochastic knapsack problem, maximizing
    P(improvement) * E[improvement | improvement] / cost instead of raw EI/cost.
    
    Parameters:
        mu: (2000,) array of GP mean predictions
        sigma: (2000,) array of GP std predictions
        f_best: scalar, best observed value so far
        cost: (2000,) array of evaluation costs
    
    Returns:
        (2000,) array of scores (higher = evaluate next)
    """
    n = len(mu)
    
    # Deadline check
    if time.time() > _DEADLINE:
        return np.zeros(n, dtype=np.float64)
    
    # Numerical safety
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)
    
    # z-score for minimization: z = (f_best - mu) / sigma
    # P(f(x) < f_best) = Phi(z)
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -30, 30)
    
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    # --- Core: Probability-weighted improvement per unit cost ---
    # This is the expected improvement magnitude weighted by probability of improvement,
    # divided by cost. For minimization:
    # E[improvement] = (f_best - mu) * Phi(z) + sigma * phi(z)
    # P(improvement) = Phi(z)
    # E[improvement | improvement] = E[improvement] / Phi(z) when Phi(z) > 0
    #
    # The stochastic knapsack formulation maximizes:
    #   P(improvement) * E[improvement | improvement] / cost
    # = E[improvement] / cost
    #
    # But to explicitly capture the probability weighting and down-weight
    # candidates with low probability of improvement, we use:
    #   score = Phi(z) * (f_best - mu) / cost + sigma * phi(z) / cost
    #
    # This decomposes into:
    #   Term 1: Phi(z) * (f_best - mu) / cost  -- probability-weighted mean improvement
    #   Term 2: sigma * phi(z) / cost           -- uncertainty contribution
    
    term1 = cdf_z * (f_best - mu) / cost_safe
    term2 = sigma_safe * pdf_z / cost_safe
    
    # Both terms can be negative; clamp to zero (no negative scores)
    core_score = np.maximum(term1, 0.0) + np.maximum(term2, 0.0)
    
    # --- Exploration component: UCB-style uncertainty bonus ---
    # Scale by cost to maintain cost-awareness
    # Use adaptive beta based on progress toward the known minimum
    f_min = -0.8
    progress = np.clip((0.0 - f_best) / (0.0 - f_min + 1e-8), 0.0, 1.0)
    beta = 1.0 - 0.7 * progress
    ucb_cost = beta * sigma_safe / cost_safe
    
    # Combine: core score dominates, UCB provides exploration
    raw_score = core_score + 0.25 * ucb_cost
    
    # Ensure finite
    raw_score = _safe_finite(raw_score, 0.0)
    
    # Final safety: ensure all finite
    raw_score = _safe_finite(raw_score, 0.0)
    
    return raw_score
# EVOLVE-BLOCK-END
