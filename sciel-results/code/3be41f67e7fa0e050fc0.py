"""Cost-aware acquisition function for Bayesian optimization (shekel 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _norm_cdf(x):
    """Vectorized standard normal CDF using erf from math (fast via numpy)."""
    # Use the relation: Phi(x) = 0.5 * (1 + erf(x / sqrt(2)))
    # But for large |x|, erf saturates. We'll use a piecewise approach.
    x = np.asarray(x, dtype=np.float64)
    result = np.empty_like(x)
    
    # For |x| > 8, CDF is essentially 0 or 1
    mask_large_pos = x > 8.0
    mask_large_neg = x < -8.0
    mask_mid = ~(mask_large_pos | mask_large_neg)
    
    result[mask_large_pos] = 1.0
    result[mask_large_neg] = 0.0
    
    if np.any(mask_mid):
        # Use scipy if available, otherwise approximate
        try:
            from scipy.special import erf
            result[mask_mid] = 0.5 * (1.0 + erf(x[mask_mid] / np.sqrt(2.0)))
        except ImportError:
            # Abramowitz and Stegun approximation for erf
            xm = x[mask_mid]
            sign = np.sign(xm)
            ax = np.abs(xm)
            t = 1.0 / (1.0 + 0.3275911 * ax)
            y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
                       - 0.284496736) * t + 0.254829592) * t * np.exp(-ax * ax)
            result[mask_mid] = 0.5 * (1.0 + sign * y)
    
    return result


def _norm_pdf(x):
    """Vectorized standard normal PDF."""
    x = np.asarray(x, dtype=np.float64)
    return np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for minimizing shekel 4d.
    
    Uses a lower-bound surrogate: maximize P(f(x) < f_best - eps) 
    which is Phi((f_best - eps - mu) / sigma), normalized by cost.
    This focuses on high-confidence improvements over a fixed threshold.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean predictions (in objective units).
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions (in objective units).
    f_best : float
        Lowest observed function value.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    t0 = time.time()
    DEADLINE = 320.0  # seconds, leave margin before 330s wall-clock
    
    # Handle edge cases
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-9)
    
    # --- Lower-bound surrogate: Probability of significant improvement ---
    # Maximize P(f(x) < f_best - eps) = Phi((f_best - eps - mu) / sigma)
    # This focuses on high-confidence improvements over a fixed threshold
    typical_scale = max(abs(f_best), 0.1)
    eps = 0.05 * typical_scale
    target = f_best - eps
    
    z_prob = (target - mu) / sigma_safe
    z_prob_clipped = np.clip(z_prob, -50, 50)
    prob_improve = _norm_cdf(z_prob_clipped)
    prob_improve = np.maximum(prob_improve, 0.0)
    
    # --- Cost-awareness: divide by cost with power < 1 ---
    # Use a softer cost penalty to allow some exploration
    scores = prob_improve / np.power(cost_safe, 0.5)
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Deadline check
    if time.time() - t0 > DEADLINE:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
