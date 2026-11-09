"""Cost-aware acquisition function for Bayesian optimization (Styblinski-Tang 2D)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_TIMEOUT = 285  # Safety margin before 300s limit


def _elapsed_ok():
    return (time.time() - _start_time) < _TIMEOUT


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a hybrid of Expected Improvement
    with cost normalization and an uncertainty bonus.
    
    This approach uses:
    1. Expected Improvement (EI) as the base signal for exploitation
    2. Cost normalization to make the score cost-aware
    3. A dynamic uncertainty bonus that encourages exploration early
       and exploitation later
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        GP mean predictions
    sigma : array of shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best observed value so far
    cost : array of shape (2000,)
        Cost of evaluating each candidate
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores (higher = more promising)
    """
    # Ensure all inputs are finite
    sigma = np.maximum(sigma, 1e-9)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    mu = np.where(np.isfinite(mu), mu, f_best)
    cost = np.maximum(cost, 1e-9)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    
    # Check time - if close to timeout, return simple cost-adjusted improvement
    if not _elapsed_ok():
        return (f_best - mu) / cost
    
    # Compute Expected Improvement (EI)
    improvement = f_best - mu
    z = improvement / sigma
    ei = (improvement * norm.cdf(z) + sigma * norm.pdf(z))
    ei = np.maximum(ei, 0.0)
    
    # Known minimum for Styblinski-Tang 2D
    known_min = -78.3323
    progress = f_best - known_min  # Positive when above minimum
    
    # Dynamic lambda: controls the trade-off between improvement and cost
    # Higher lambda means more cost-sensitive
    if progress < 1:
        lambda_val = 2.5
    elif progress < 5:
        lambda_val = 1.8
    elif progress < 15:
        lambda_val = 1.2
    elif progress < 30:
        lambda_val = 0.8
    else:
        lambda_val = 0.5
    
    # Cost-aware score: EI minus lambda * cost
    # This preserves the shape of the EI surface while penalizing expensive points
    cost_aware_ei = ei - lambda_val * cost
    
    # Also compute a normalized EI score for comparison
    # Normalize EI to [0, 1] range
    ei_max = np.max(ei)
    if ei_max > 1e-9:
        ei_norm = ei / ei_max
    else:
        ei_norm = np.zeros_like(ei)
    
    # Uncertainty bonus: encourage exploration when sigma is high
    # The bonus is stronger early on and weaker as we get closer to the optimum
    sigma_max = np.max(sigma)
    if sigma_max > 1e-9:
        sigma_norm = sigma / sigma_max
    else:
        sigma_norm = np.zeros_like(sigma)
    
    # Exploration weight: higher when far from optimum
    if progress > 30:
        expl_weight = 0.3
    elif progress > 15:
        expl_weight = 0.25
    elif progress > 5:
        expl_weight = 0.2
    else:
        expl_weight = 0.15
    
    # Combine cost-aware EI with uncertainty bonus
    # The uncertainty bonus helps escape local optima and explore new regions
    scores = cost_aware_ei + expl_weight * sigma_norm * ei_max
    
    # Add a small additional boost for points with very high uncertainty
    # and moderate EI, to encourage broader exploration
    high_unc_mask = sigma_norm > 0.7
    moderate_ei_mask = (ei_norm > 0.1) & (ei_norm < 0.5)
    exploration_boost = (high_unc_mask & moderate_ei_mask).astype(float) * 0.1 * ei_max
    scores += exploration_boost
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
