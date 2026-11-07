"""Cost-aware acquisition function for Bayesian optimization (ackley 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition score for 2000 candidates.
    Higher score means more preferred to evaluate next.
    Uses Expected Improvement (EI) divided by evaluation cost.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Ensure cost is positive to avoid division by zero
    cost_safe = np.maximum(cost, 1e-9)

    # Calculate z-score for Expected Improvement
    # EI(x) = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    improvement = f_best - mu
    z = improvement / sigma_safe
    
    # Clip z for numerical stability
    z = np.clip(z, -10.0, 10.0)

    cdf = norm.cdf(z)
    pdf = norm.pdf(z)

    # Standard expected improvement
    ei = improvement * cdf + sigma_safe * pdf
    
    # Ensure EI is non-negative (numerical stability)
    ei = np.maximum(ei, 0.0)

    # Cost-aware: divide by cost
    scores = ei / cost_safe

    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
# EVOLVE-BLOCK-END
