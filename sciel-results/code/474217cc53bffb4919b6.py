"""Evolved program: efficient cost-aware acquisition (EI / cost) with fast CDF."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_DEADLINE = time.time() + 320.0  # safety margin before the 330 s kill
_EPS = 1e-9
_SQRT_2PI = np.sqrt(2.0 * np.pi)


def _fast_cdf(z):
    """Vectorized standard normal CDF using math.erf via numpy."""
    # Use scipy's norm.cdf which is vectorized and fast
    return norm.cdf(z)


def _fast_pdf(z):
    """Vectorized standard normal PDF."""
    return np.exp(-0.5 * z * z) / _SQRT_2PI


def _expected_improvement(mu, sigma, f_best):
    """Analytic expected improvement (lower-is-better) per candidate."""
    s = np.maximum(sigma, _EPS)
    z = (f_best - mu) / s
    cdf = _fast_cdf(z)
    pdf = _fast_pdf(z)
    ei = (f_best - mu) * cdf + s * pdf
    return np.maximum(ei, 0.0)


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition: EI / cost.
    
    Parameters
    ----------
    mu : array-like, shape (2000,)
        GP mean predictions.
    sigma : array-like, shape (2000,)
        GP standard deviation predictions.
    f_best : float
        Best observed function value.
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate.
    
    Returns
    -------
    scores : ndarray, shape (2000,)
        Finite scores (higher = evaluate next).
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure finite inputs
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, _EPS)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    
    # Clip to safe ranges
    sigma = np.maximum(sigma, _EPS)
    cost = np.maximum(cost, _EPS)
    
    # Compute EI
    ei = _expected_improvement(mu, sigma, f_best)
    
    # Cost-aware score: EI per unit cost
    scores = ei / cost
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Add small exploration bonus based on uncertainty
    # This helps when EI is zero (mu >= f_best everywhere)
    # Use a small UCB-like term to break ties
    ucb_bonus = 0.1 * sigma / cost
    scores = scores + ucb_bonus
    
    # Final safety: ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
