"""Cost-aware acquisition function for Bayesian optimization (levy 2d)."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

# Per-process deadline guard
_T0 = time.time()
_DEADLINE = 330.0

def _elapsed():
    return time.time() - _T0

def acquisition(mu, sigma, f_best, cost):
    # Guard: if we are dangerously close to the wall-clock limit, return a
    # deterministic, finite, low-risk signal immediately.
    if _elapsed() > _DEADLINE - 1.0:
        return np.zeros(np.shape(mu), dtype=float)

    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)
    
    # Sanitize inputs
    # sigma should be positive. GP noise is 1e-6, so sigma shouldn't be 0, but just in case.
    sigma = np.clip(sigma, 1e-6, None)
    cost = np.clip(cost, 1e-6, None)
    f_best = float(f_best) if np.isfinite(f_best) else 0.0

    # Calculate Expected Improvement (for minimization)
    # EI(x) = E[max(f_best - f(x), 0)]
    # = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    
    z = (f_best - mu) / sigma
    # Clip z to avoid extreme values that might cause numerical issues in cdf/pdf
    # although scipy is robust, clipping to [-10, 10] is safe.
    z = np.clip(z, -10.0, 10.0)
    
    # Compute EI
    # Using standard normal CDF and PDF
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_z + sigma * pdf_z
    
    # Cost-aware score: EI / cost
    # We want to maximize this score.
    score = ei / cost
    
    # Exploration bonus: bounded by sigma to prevent dominance
    # Using sigma / (sigma + 0.1) as a smooth saturation function
    # This encourages exploration in uncertain regions without overwhelming exploitation
    # The 0.1 scaling factor balances exploitation (EI/cost) with exploration
    exploration_bonus = 0.1 * sigma / (sigma + 0.1)
    
    # Final score: combine EI/cost with exploration bonus
    score = score + exploration_bonus
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
