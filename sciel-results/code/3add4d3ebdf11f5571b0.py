"""Cost-aware acquisition for three hump camel 2d BO loop."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

# Precompute nothing heavy; keep state minimal.
_deadline = time.time() + 320.0  # safety margin under 330s kill
_initialized = False
_init_time = None


def _ensure_init():
    global _initialized, _init_time
    if not _initialized:
        _init_time = time.time()
        _initialized = True


def acquisition(mu, sigma, f_best, cost):
    """
    mu, sigma, cost: arrays of shape (2000,)
    Returns 2000 finite scores (higher = evaluate next).
    """
    _ensure_init()

    # Deadline guard
    if time.time() > _deadline:
        # Return a simple, fast fallback
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        return ei / (cost + 1e-9)

    # Ensure inputs are numpy arrays of float64 for consistent performance
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    f_best = float(f_best)

    # Clip sigma to avoid division by zero or negative values
    # Use a small epsilon to prevent numerical instability
    sigma_safe = np.maximum(sigma, 1e-8)

    # Ensure cost is positive
    cost_safe = np.maximum(cost, 1e-8)

    # Calculate z-score for Expected Improvement
    z = (f_best - mu) / sigma_safe

    # Calculate Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z

    # EI should be non-negative, but numerical errors can make it slightly negative
    # Clamp to 0 to ensure we only consider positive improvements
    ei = np.maximum(ei, 0.0)

    # Cost-aware: divide by cost.
    # The finding indicates that strict linear cost penalty (EI/Cost) is effective.
    # This prioritizes high-efficiency evaluations (improvement per unit cost).
    score = ei / cost_safe

    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
