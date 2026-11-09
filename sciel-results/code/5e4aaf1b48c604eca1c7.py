"""Cost-aware acquisition function with Cost-Aware EI for Powell 4D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE_MARGIN = 3.0  # seconds before the hard 330s limit


def _elapsed():
    return time.time() - _START_TIME


def _deadline_exceeded():
    return _elapsed() > (330.0 - _DEADLINE_MARGIN)


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization (powell 4d).

    Uses Cost-Aware Expected Improvement (CAEI). The score is
    Expected Improvement / cost.
    
    EI = (f_best - mu) * Phi((f_best - mu) / sigma) + sigma * phi((f_best - mu) / sigma)
    
    This directly optimizes the expected reduction in simple regret per unit cost.

    Parameters
    ----------
    mu : np.ndarray, shape (2000,)
        GP mean predictions in objective units.
    sigma : np.ndarray, shape (2000,)
        GP standard deviation predictions in objective units.
    f_best : float
        Lowest observed function value so far.
    cost : np.ndarray, shape (2000,)
        Cost of evaluating each candidate, shape (2000,).

    Returns
    -------
    np.ndarray, shape (2000,)
        Finite scores; higher = prefer to evaluate next.
    """
    # Explicit deadline guard
    if _deadline_exceeded():
        # Return a safe, finite fallback
        sigma_safe = np.maximum(sigma, 1e-9)
        # Fallback to LCB if time is up
        score = (f_best - mu + 0.5 * sigma_safe) / (cost + 1e-9)
        return np.where(np.isfinite(score), score, -1e10)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Guard against non-finite or invalid values
    mu = np.where(np.isfinite(mu), mu, 1e10)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)

    sigma_safe = np.maximum(sigma, 1e-9)

    # Calculate Z-score for EI
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues in norm.cdf/pdf for extreme values
    # norm.cdf(-10) is ~0, norm.pdf(-10) is ~0.
    # We can clip to [-10, 10] for stability, though scipy handles it well.
    z_clipped = np.clip(z, -10.0, 10.0)
    
    # Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Note: (f_best - mu) = z * sigma
    # So EI = sigma * (z * Phi(z) + phi(z))
    # This form is numerically more stable as it avoids large/small multiplication errors if f_best-mu is huge?
    # Actually, if f_best - mu is huge, z is huge, Phi(z) is 1, phi(z) is 0. EI ~ f_best - mu.
    # Using sigma * (z*Phi + phi) is equivalent.
    
    ei = sigma_safe * (z_clipped * norm.cdf(z_clipped) + norm.pdf(z_clipped))
    
    # Cost-Aware EI
    score = ei / cost
    
    # Ensure finiteness
    score = np.where(np.isfinite(score), score, -1e10)

    return score
# EVOLVE-BLOCK-END
