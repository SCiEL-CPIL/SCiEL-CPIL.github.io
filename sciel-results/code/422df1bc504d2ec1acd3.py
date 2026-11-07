"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
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
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses Probability of Improvement (PI) per unit cost as the primary signal.
    PI = Phi((f_best - mu) / sigma) gives the probability that f(x) < f_best.
    Scaling by 1/cost creates a cost-aware probability metric that favors
    cheap evaluations with high likelihood of improvement.
    
    This shifts the objective from maximizing expected magnitude of improvement
    (EI) to maximizing the likelihood of finding any improvement, which may
    better navigate the Levy function landscape by encouraging exploration
    of promising regions without over-committing to high-magnitude predictions.
    """
    # Guard: if we are dangerously close to the wall-clock limit, return a
    # deterministic, finite, low-risk signal immediately.
    if _elapsed() > _DEADLINE - 1.0:
        return np.zeros(np.shape(mu), dtype=float)

    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)

    # Sanitize inputs
    sigma = np.clip(sigma, 1e-9, None)
    cost = np.clip(cost, 1e-9, None)
    f_best = float(f_best) if np.isfinite(f_best) else 0.0

    # Calculate Probability of Improvement (for minimization)
    # PI(x) = P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma
    # Clip z to avoid extreme values that might cause numerical issues
    z = np.clip(z, -10.0, 10.0)
    
    # Compute PI
    pi = norm.cdf(z)
    
    # Cost-aware score: PI / cost
    # We want to maximize this score.
    score = pi / cost
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
