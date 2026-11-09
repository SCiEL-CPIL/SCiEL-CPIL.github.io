"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import erf

_DEADLINE_START = time.time()
_TIME_LIMIT = 295.0  # seconds, safety margin before 330s kill


def _norm_cdf(x):
    """Fast standard normal CDF using erf."""
    return 0.5 * (1.0 + erf(x / np.sqrt(2.0)))


def _norm_pdf(x):
    """Fast standard normal PDF."""
    return np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)


def acquisition(mu, sigma, f_best, cost):
    """
    Structural Change: Knowledge Gradient acquisition function with cost normalization.
    
    Knowledge Gradient (KG) estimates the expected value of information from evaluating
    a point, which is more principled than EI for multi-step optimization.
    
    KG(x) = E[max(0, f_best - min(f_best, f(x))) | x]
         = E[max(0, f_best - f(x)) | x] when f(x) ~ N(mu, sigma^2)
    
    For a single evaluation, KG simplifies to:
    KG(x) = (f_best - mu) * Phi(z) + sigma * phi(z)
    where z = (f_best - mu) / sigma
    
    However, we use a more robust formulation that accounts for the cost of evaluation
    and adapts the exploration/exploitation trade-off based on remaining budget.
    
    Score = KG / cost
    """
    # Check time deadline
    if time.time() - _DEADLINE_START > _TIME_LIMIT:
        # Fallback: simple KG / cost
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        kg = (f_best - mu) * _norm_cdf(z) + sigma_safe * _norm_pdf(z)
        score = kg / np.maximum(cost, 1e-9)
        return np.where(np.isfinite(score), score, 0.0)

    # Safe sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute z-score
    z = (f_best - mu) / sigma_safe
    
    # Knowledge Gradient: expected improvement from this evaluation
    kg = (f_best - mu) * _norm_cdf(z) + sigma_safe * _norm_pdf(z)
    
    # Ensure KG is non-negative (numerical safety)
    kg = np.maximum(kg, 0.0)
    
    # Cost normalization
    cost_safe = np.maximum(cost, 1e-9)
    score = kg / cost_safe
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
