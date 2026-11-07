"""Cost-aware acquisition function for Bayesian optimization (cosine8 8d)."""
import numpy as np
from scipy.stats import norm
import time

# Global timer for safety
_START_TIME = time.time()
_TIME_LIMIT = 295.0  # Safety margin below 330s wall-clock limit

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        GP mean predictions
    sigma : array-like, shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best observed value so far
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next)
    """
    # Explicit deadline guard
    elapsed = time.time() - _START_TIME
    if elapsed > _TIME_LIMIT:
        # Fallback: simple cost-aware EI
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        cost_safe = np.maximum(cost, 1e-9)
        return ei / cost_safe
    
    # Convert to numpy arrays for efficiency
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute z-scores
    z = (f_best - mu) / sigma_safe
    
    # Clip z for numerical stability
    z_clipped = np.clip(z, -10, 10)
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    
    # Ensure EI is non-negative (numerical issues can cause tiny negative values)
    ei = np.maximum(ei, 0.0)
    
    # Cost normalization: cost is 1 + mean(x_scaled) where x_scaled in [0,1]
    # So cost is in [1, 2].
    cost_safe = np.maximum(cost, 1e-9)
    
    # Structural change: Use a hybrid acquisition that combines:
    # 1. Expected Improvement (exploitation)
    # 2. Uncertainty-based exploration (sigma)
    # 3. Cost-awareness via a power penalty
    
    # The key insight: cosine8 has a global minimum at the center of the domain.
    # Cost is 1 + mean(x_scaled), so center points have lower cost (~1.0) while
    # boundary points have higher cost (~2.0).
    
    # Use a cost-aware EI with a stronger cost penalty to favor cheaper points
    # that are likely to be near the center where the minimum is.
    
    # Compute cost-aware EI
    ei_cost = ei / (cost_safe ** 1.5)
    
    # Add an exploration bonus that is also cost-aware
    # The exploration term encourages visiting uncertain regions but penalizes expensive ones
    max_sigma = np.max(sigma_safe)
    exploration = sigma_safe / (cost_safe ** 1.5)
    
    # Combine: primarily EI, with a moderate exploration component
    # The exploration helps when EI is low (e.g., early in optimization or in flat regions)
    scores = ei_cost + 0.1 * exploration
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
