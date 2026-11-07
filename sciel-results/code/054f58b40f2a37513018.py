"""Cost-aware acquisition for Griewank-2D BO with deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Wall-clock deadline guard (330 s kill, use 325 s safety margin)
_T0 = time.time()
_DEADLINE = 325.0


def _remaining():
    """Return seconds remaining before deadline; 0 if exceeded."""
    return max(0.0, _DEADLINE - (time.time() - _T0))


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition score for minimizing Griewank-2D.

    Uses Expected Improvement (EI) combined with an uncertainty bonus
    (proportional to predictive std) to promote diversity and escape
    local optima. Both terms are normalized by evaluation cost.

    Parameters
    ----------
    mu : ndarray, shape (2000,)
        GP posterior mean in objective units.
    sigma : ndarray, shape (2000,)
        GP posterior std in objective units.
    f_best : float
        Best (lowest) observed objective value.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate.

    Returns
    -------
    ndarray, shape (2000,)
        Finite scores; higher = more worth evaluating next.
    """
    # Guard: if we're near deadline, return a safe finite array immediately
    if _remaining() < 0.5:
        return np.zeros_like(mu, dtype=np.float64)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Clamp to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-10)

    # z-score for EI computation
    z = (f_best - mu) / sigma_safe

    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is CDF and phi is PDF of standard normal
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z

    # Clamp EI to be non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)

    # Uncertainty bonus: encourages exploration of high-variance regions
    # This promotes diversity and helps escape local optima in Griewank
    beta = 0.5  # Exploration-exploitation balance
    uncertainty_bonus = beta * sigma_safe

    # Combined score: (EI + uncertainty bonus) / cost
    score = (ei + uncertainty_bonus) / cost_safe

    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)

    return score
# EVOLVE-BLOCK-END
