"""Cost-aware acquisition function for Bayesian optimization (levy 2d)."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

# Per-process deadline guard
_T0 = time.time()
_DEADLINE = 300.0
_SAFETY_MARGIN = 5.0

def _elapsed():
    return time.time() - _T0

def acquisition(mu, sigma, f_best, cost):
    # Guard: if we are dangerously close to the wall-clock limit, return a
    # deterministic, finite, low-risk signal immediately.
    if _elapsed() > _DEADLINE - _SAFETY_MARGIN:
        # Fallback: simple EI/cost
        sigma_safe = np.maximum(np.asarray(sigma, dtype=float), 1e-8)
        cost_safe = np.maximum(np.asarray(cost, dtype=float), 1e-8)
        f_best = float(f_best) if np.isfinite(f_best) else 0.0
        z = (f_best - np.asarray(mu, dtype=float)) / sigma_safe
        z = np.clip(z, -10.0, 10.0)
        ei = (f_best - np.asarray(mu, dtype=float)) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        scores = ei / cost_safe
        return np.where(np.isfinite(scores), scores, 0.0)

    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)
    
    # Sanitize inputs
    sigma = np.clip(sigma, 1e-6, None)
    cost = np.clip(cost, 1e-6, None)
    f_best = float(f_best) if np.isfinite(f_best) else 0.0

    # Calculate Expected Improvement (for minimization)
    # EI(x) = E[max(f_best - f(x), 0)]
    # = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    
    z = (f_best - mu) / sigma
    # Clip z to avoid extreme values that might cause numerical issues in cdf/pdf
    z = np.clip(z, -10.0, 10.0)
    
    # Compute EI
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_z + sigma * pdf_z
    
    # Cost-aware score: EI / cost
    # We want to maximize this score.
    score = ei / cost
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
