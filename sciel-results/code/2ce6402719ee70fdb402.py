"""Acquisition function for Bayesian optimization of Hartmann 6D."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global deadline guard
_START_TIME = time.time()
# The evaluator kills after 330s. We use a safety margin of 10s to ensure 
# we return a valid result before the hard kill.
_DEADLINE = _START_TIME + 320.0


def _check_deadline():
    """Return True if we are past the safety deadline."""
    return time.time() > _DEADLINE


def acquisition(mu, sigma, f_best):
    """
    Acquisition function: returns 2000 finite scores.
    Higher score = more likely to be evaluated next.
    
    Uses a hybrid acquisition function combining Expected Improvement (EI)
    with a Upper Confidence Bound (UCB) exploration term.
    
    EI focuses on exploitation (finding points likely to be better than f_best).
    UCB focuses on exploration (finding points with high uncertainty).
    In high-dimensional spaces like Hartmann 6D, pure EI can get stuck in local
    minima or over-exploit the GP mean. Adding a UCB term helps escape these
    traps by rewarding regions where the GP is uncertain.
    """
    # Deadline guard: If we are running out of time, return a simple exploitative
    # score to ensure we don't crash or timeout. -mu is a safe fallback that
    # prioritizes the best predicted points.
    if _check_deadline():
        safe_mu = np.where(np.isfinite(mu), mu, f_best)
        # Ensure we return finite values
        return -safe_mu

    # Sanitize inputs
    # sigma must be positive for EI calculation
    # Clip sigma to a small positive value to avoid division by zero
    # and to handle cases where the GP is very certain (sigma ~ 0)
    sigma = np.maximum(sigma, 1e-10)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-10)
    
    # mu should be finite. If not, treat it as very bad (high value for minimization)
    # or replace with f_best to make EI 0. Replacing with f_best is safer for stability.
    mu = np.where(np.isfinite(mu), mu, f_best)
    
    # Expected Improvement for minimization
    # We use a small exploration parameter xi to slightly shift the improvement
    # threshold, encouraging exploration of points that are slightly worse than
    # the current best on average but have high potential.
    xi = 0.01
    
    # z = (f_best - mu - xi) / sigma
    z = (f_best - mu - xi) / sigma
    
    # Clip z to a reasonable range to avoid numerical issues in extreme tails
    # Phi(z) -> 1 for z >> 0, Phi(z) -> 0 for z << 0
    # phi(z) -> 0 for |z| >> 0
    # Clipping to [-10, 10] is safe and standard.
    z = np.clip(z, -10, 10)
    
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    # EI = (f_best - mu - xi) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu - xi) * cdf_z + sigma * pdf_z
    
    # UCB Exploration Term
    # This term rewards points with high uncertainty (high sigma).
    # The coefficient 0.1 is a balance factor. It is small enough that EI dominates
    # in low-uncertainty regions (exploitation) but large enough to guide the search
    # towards unexplored areas (exploration) when the GP is uncertain.
    ucb_boost = 0.1 * sigma
    
    # Combine EI and UCB boost
    scores = ei + ucb_boost
    
    # Ensure all scores are finite and non-negative (EI should be >= 0, UCB is >= 0)
    scores = np.maximum(scores, 0.0)
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
