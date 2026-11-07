"""Cost-aware acquisition for Bayesian optimization of powell 4d (min 0) on [-4,5]^4."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

_EPS = 1e-9
_DEADLINE_S = 330.0
_SAFETY_S = 2.0
# Exploration weight for the sigma term. 
# Placed inside the cost division to be cost-aware.
_EXPLORE_WEIGHT = 1.0


def acquisition(mu, sigma, f_best, cost):
    t0 = time.time()
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Guard against non-finite inputs
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, _EPS)
    sigma = np.clip(sigma, _EPS, None)
    cost = np.where(np.isfinite(cost), cost, 1.5)
    cost = np.clip(cost, _EPS, None)

    # Expected Improvement (EI)
    # improvement = f_best - mu
    # z = improvement / sigma
    # EI = improvement * cdf(z) + sigma * pdf(z)
    improvement = f_best - mu
    z = improvement / sigma
    
    # Clip z to avoid numerical issues in cdf/pdf
    z = np.clip(z, -50, 50)
    
    ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure EI is non-negative (numerical stability)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score
    # We combine EI and an exploration bonus (sigma) and divide by cost.
    # This encourages finding improvements in cheap regions, and exploring uncertain cheap regions.
    score = (ei + _EXPLORE_WEIGHT * sigma) / cost

    # Ensure finiteness
    score = np.where(np.isfinite(score), score, -1e30)

    # Deadline guard
    if time.time() - t0 > (_DEADLINE_S - _SAFETY_S):
        return np.where(np.isfinite(score), score, 0.0)

    return np.where(np.isfinite(score), score, 0.0)
# EVOLVE-BLOCK-END
