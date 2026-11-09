import numpy as np
from scipy.special import erfc, erf
import time

# Module-level deadline guard
_acq_start_time = None
_DEADLINE = 325.0  # 5s safety margin before 330s kill

def _check_deadline():
    """Return True if we should stop and return quickly."""
    if _acq_start_time is None:
        return False
    return (time.monotonic() - _acq_start_time) > _DEADLINE

def _fast_norm_cdf(z):
    """Vectorized standard normal CDF using erfc for speed."""
    return 0.5 * erfc(-z / np.sqrt(2.0))

def _fast_norm_pdf(z):
    """Vectorized standard normal PDF."""
    return np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses Expected Improvement (EI) with exponential cost penalty to suppress
    high-cost regions, plus a small exploration bonus.
    
    Parameters:
    - mu: array of shape (2000,) with GP mean predictions
    - sigma: array of shape (2000,) with GP standard deviation predictions
    - f_best: float, the best (lowest) observed value so far
    - cost: array of shape (2000,) with evaluation costs
    
    Returns:
    - array of shape (2000,) with acquisition scores (higher = better)
    """
    global _acq_start_time
    _acq_start_time = time.monotonic()
    
    # Ensure sigma is non-negative and add a small epsilon to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Compute z-scores: how many standard deviations below f_best the mean is
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is the CDF and phi is the PDF of the standard normal
    ei = (f_best - mu) * _fast_norm_cdf(z) + sigma_safe * _fast_norm_pdf(z)
    
    # Ensure EI is non-negative (it should be, but clip for safety)
    ei = np.maximum(ei, 0)
    
    # Exponential cost penalty: exp(-beta * cost) to suppress high-cost regions
    # Cost ranges from 1 to 2 (1 + mean(x scaled to [0,1]))
    # beta = 1.0 gives penalty range [exp(-1), exp(-2)] ≈ [0.37, 0.135]
    # This exponentially penalizes expensive evaluations
    beta = 1.0
    cost_penalty = np.exp(-beta * cost)
    
    # Cost-aware EI: multiply by exponential penalty
    cost_aware_ei = ei * cost_penalty
    
    # Small exploration bonus to help escape local minima
    # Scale by cost penalty as well to maintain cost awareness
    exploration_weight = 0.05
    exploration_bonus = exploration_weight * sigma_safe * cost_penalty
    
    # Final score: combine cost-aware EI with exploration bonus
    score = cost_aware_ei + exploration_bonus
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
