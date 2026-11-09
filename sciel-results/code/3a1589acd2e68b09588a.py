"""Cost-aware acquisition: model-and-optimize EI with learned exploration.
Uses Expected Improvement (EI) divided by cost, with kappa adapted based on
the state of the GP model. EI is more principled than UCB for minimization
and the cost normalization handles the variable evaluation cost.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# --- Explicit deadline guard (330 s hard kill; keep a safety margin) ---
_START_TIME = time.time()
_DEADLINE = _START_TIME + 320.0   # 10 s safety margin before the 330 s kill

def _elapsed_ok():
    """True while we have at least ~3 s of margin before the hard limit."""
    return time.time() < _DEADLINE - 3.0

def _ei(mu, sigma, f_best):
    """Compute Expected Improvement for minimization."""
    improvement = f_best - mu
    sigma_safe = np.maximum(sigma, 1e-10)
    z = improvement / sigma_safe
    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    return np.maximum(ei, 0.0)

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition for Bayesian optimization (minimization).

    mu, sigma, f_best, cost : each shape (2000,)
      f_best : scalar, lowest observed value
      cost   : per-candidate cost = 1 + mean(x_scaled)  (in [1, 2])
    Returns 2000 finite scores; higher = evaluate next.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    f_best = float(f_best)

    # Robust floors -------------------------------------------------------
    cost_safe = np.maximum(cost, 1e-9)
    sigma_safe = np.maximum(sigma, 1e-10)

    # Deadline guard: fall back to a minimal valid score if near the limit.
    if not _elapsed_ok():
        score = np.zeros_like(mu)
        return score

    # Compute base EI
    ei_base = _ei(mu, sigma_safe, f_best)
    
    # Adapt kappa based on uncertainty: use higher kappa when sigma is large
    # relative to the improvement, to encourage exploration
    sigma_mean = float(np.mean(sigma_safe))
    improvement = f_best - mu
    improvement_mean = float(np.mean(np.maximum(improvement, 0)))
    
    # If there's still significant uncertainty relative to improvement, explore more
    if improvement_mean > 1e-6:
        uncertainty_ratio = sigma_mean / improvement_mean
    else:
        uncertainty_ratio = 1.0
    
    # Map uncertainty ratio to kappa: more uncertainty -> higher kappa
    # Typical range: 1.0 to 4.0
    kappa = np.clip(1.0 + 3.0 * uncertainty_ratio, 1.0, 4.0)
    
    # Compute EI with kappa adjustment (use kappa to scale the exploration term)
    # EI_kappa = (f_best - mu) * Phi(z) + kappa * sigma * phi(z)
    z = improvement / sigma_safe
    ei_kappa = improvement * norm.cdf(z) + kappa * sigma_safe * norm.pdf(z)
    ei_kappa = np.maximum(ei_kappa, 0.0)
    
    # Cost-aware score: EI / cost
    score = ei_kappa / cost_safe
    
    # Numerical safety: finite, non-negative.
    score = np.where(np.isfinite(score), score, 0.0)
    score = np.maximum(score, 0.0)

    return score
# EVOLVE-BLOCK-END
