"""Cost-aware acquisition function for Bayesian optimization (cosine8 8d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import erfc, log_ndtr
import time

# Deadline guard: total wall-clock limit is 330s, keep safety margin
_DEADLINE = 330.0
_START_TIME = None


def _ensure_start():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return _START_TIME


def _log_norm_cdf(z):
    """Log of standard normal CDF, numerically stable."""
    # For large positive z, log Phi(z) ~ 0
    # For large negative z, log Phi(z) ~ log(1 - Phi(-z)) ~ log(1 - phi(z)/|z|)
    # Use erfc for the tail
    # Phi(z) = 0.5 * erfc(-z / sqrt(2))
    # For z > 0: use log(0.5 * erfc(-z/sqrt(2)))
    # For z <= 0: use log_ndtr from scipy or manual
    z = np.asarray(z, dtype=np.float64)
    result = np.empty_like(z)
    pos = z > 0
    neg = ~pos
    if np.any(pos):
        # For positive z, Phi(z) is close to 1 for large z
        # log Phi(z) = log(1 - Phi(-z))
        # For large positive z, Phi(-z) is tiny, so log(1 - tiny) ~ -tiny
        # Better: use the fact that for z > 0, log Phi(z) = log(1 - 0.5*erfc(z/sqrt(2)))
        # Actually, let's just use the direct formula with erfc
        # Phi(z) = 0.5 * erfc(-z / sqrt(2))
        val = 0.5 * erfc(-z[pos] / np.sqrt(2.0))
        result[pos] = np.log(np.maximum(val, 1e-300))
    if np.any(neg):
        # For negative z, Phi(z) = 0.5 * erfc(-z / sqrt(2)) where -z is positive
        # So this is the small tail
        val = 0.5 * erfc(-z[neg] / np.sqrt(2.0))
        result[neg] = np.log(np.maximum(val, 1e-300))
    return result


def _log_norm_pdf(z):
    """Log of standard normal PDF."""
    return -0.5 * z * z - 0.5 * np.log(2.0 * np.pi)


def _expected_improvement(mu, sigma, f_best):
    """
    Compute Expected Improvement for minimization.
    EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    where z = (f_best - mu) / sigma
    """
    sigma_safe = np.maximum(sigma, 1e-10)
    delta = f_best - mu  # positive when improvement possible
    z = delta / sigma_safe
    
    # Clip z for numerical stability
    z_clipped = np.clip(z, -100, 100)
    
    # Compute EI in log space for numerical stability
    # EI = delta * Phi(z) + sigma * phi(z)
    # For large positive z: EI ~ delta (since Phi(z) ~ 1, phi(z) ~ 0)
    # For large negative z: EI ~ 0 (since Phi(z) ~ 0, phi(z) ~ 0)
    
    # Direct computation with clipping should be fine for z in [-100, 100]
    from scipy.stats import norm
    ei = delta * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    
    # Ensure non-negative
    ei = np.maximum(ei, 0.0)
    
    return ei


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition: balance expected improvement with cost.
    
    Parameters:
    - mu: (2000,) mean predictions
    - sigma: (2000,) std predictions  
    - f_best: scalar, best observed value
    - cost: (2000,) evaluation costs
    
    Returns:
    - (2000,) acquisition scores (higher = evaluate next)
    """
    _ensure_start()
    
    # Safety: if we're too close to deadline, return simple fallback
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE - 5.0:
        return np.zeros_like(mu)
    
    # Clip sigma to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Compute Expected Improvement
    ei = _expected_improvement(mu, sigma_safe, f_best)
    
    # Cost-aware: divide by cost to get improvement per unit cost
    cost_safe = np.maximum(cost, 1e-10)
    
    # Use a power of cost for more aggressive cost avoidance
    # score = EI / cost^alpha where alpha > 1 penalizes expensive points more
    alpha = 1.5
    cost_penalty = np.power(cost_safe, alpha)
    score = ei / cost_penalty
    
    # Add exploration bonus for high-uncertainty regions
    # This helps when EI is zero (mu >= f_best) but sigma is large
    # Use a smaller coefficient to not over-explore
    uncertainty_bonus = 0.05 * sigma_safe / cost_safe
    
    final_score = score + uncertainty_bonus
    
    # Ensure finite values
    final_score = np.where(np.isfinite(final_score), final_score, 0.0)
    
    return final_score
# EVOLVE-BLOCK-END
