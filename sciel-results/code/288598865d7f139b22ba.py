"""Combined PoI + light exploitation for cost-aware Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import erf

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 310  # 20s safety margin before 330s kill

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition: PoI as primary signal + small exploitation bonus.
    
    score = (PoI + alpha * max(0, f_best - mu)) / cost
    
    PoI = Phi((f_best - mu) / sigma) captures likelihood of improvement.
    The exploitation bonus gently favors points already below f_best.
    Division by cost prefers cheaper evaluations.
    """
    # Deadline guard: return fast fallback
    if time.time() > _DEADLINE:
        mu = np.asarray(mu, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        cost = np.asarray(cost, dtype=np.float64)
        sigma_safe = np.maximum(sigma, 1e-8)
        z = np.clip((f_best - mu) / sigma_safe, -50, 50)
        poi = 0.5 * (1.0 + erf(z / np.sqrt(2.0)))
        score = poi / (cost + 1e-9)
        return np.where(np.isfinite(score), score, 0.0)

    # Convert inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Safe sigma
    sigma_safe = np.maximum(sigma, 1e-8)

    # Compute z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -50.0, 50.0)

    # Probability of Improvement: Phi(z)
    # Using erf for speed (avoids scipy.stats import overhead per call)
    poi = 0.5 * (1.0 + erf(z / np.sqrt(2.0)))

    # Small exploitation bonus: favor points where mu < f_best
    # This is a gentle push toward known-good regions without dominating
    exploitation = np.maximum(0.0, f_best - mu)
    alpha = 0.1  # Very small weight to keep PoI dominant

    # Combine: PoI primary, exploitation secondary
    numerator = poi + alpha * exploitation

    # Cost normalization
    score = numerator / (cost + 1e-9)

    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)

    return score
# EVOLVE-BLOCK-END
