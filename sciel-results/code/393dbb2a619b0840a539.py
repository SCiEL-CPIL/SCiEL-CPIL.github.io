"""Combined approach: Pure EI with adaptive sigma floor for Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Track global start time for deadline guard
_global_start_time = None
_TIMEOUT_MARGIN = 10.0  # seconds to spare before 330s limit
_TOTAL_LIMIT = 330.0


def _check_deadline():
    """Check if we are approaching the timeout limit."""
    global _global_start_time
    if _global_start_time is None:
        _global_start_time = time.time()
    elapsed = time.time() - _global_start_time
    return elapsed > (_TOTAL_LIMIT - _TIMEOUT_MARGIN)


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization.
    
    Uses pure Expected Improvement with an adaptive sigma floor that
    decays over time. Early on, a higher floor encourages exploration
    by preventing EI from collapsing to zero in low-uncertainty regions.
    Later, the floor shrinks so exploitation dominates.
    
    Args:
        mu: array of shape (2000,) - posterior mean
        sigma: array of shape (2000,) - posterior standard deviation
        f_best: float - lowest observed value
    
    Returns:
        array of shape (2000,) - scores, higher means evaluate next
    """
    global _global_start_time, _eval_count
    
    if _global_start_time is None:
        _global_start_time = time.time()
    
    # Check deadline - if close to timeout, return pure EI with tiny floor
    if _check_deadline():
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        return np.where(np.isfinite(ei), ei, 0.0)
    
    # Increment evaluation counter
    if not hasattr(acquisition, '_eval_count'):
        acquisition._eval_count = 0
    acquisition._eval_count += 1
    eval_count = acquisition._eval_count
    
    # Adaptive sigma floor: starts at a meaningful fraction of the problem scale,
    # decays to near-zero as we approach the end of the budget.
    # With 30 total evaluations, progress goes from ~1/30 to 30/30.
    progress = min(eval_count / 30.0, 1.0)
    
    # The RBF mixture has values roughly in [-3, 3] range based on the known min of -2.53.
    # A floor of 0.5 early on means points with sigma < 0.5 get treated as having sigma=0.5,
    # which keeps EI nonzero and encourages the optimizer to visit them.
    # By the end, the floor is ~1e-4, essentially pure EI.
    sigma_floor = 0.5 * (1.0 - progress) + 1e-4
    
    # Ensure sigma is at least the floor
    sigma_safe = np.maximum(sigma, sigma_floor)
    
    # Calculate Expected Improvement (minimization)
    # EI(x) = E[max(f_best - f(x), 0)] where f(x) ~ N(mu(x), sigma(x)^2)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure all scores are finite
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
# EVOLVE-BLOCK-END
