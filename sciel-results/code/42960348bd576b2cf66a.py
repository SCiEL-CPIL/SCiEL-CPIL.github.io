"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard: 330s wall-clock limit, use 300s for safety
_DEADLINE = 300.0
_START_TIME = None


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses a hybrid approach combining:
    1. Expected Improvement (EI) for exploitation
    2. Upper Confidence Bound (UCB) for exploration
    3. Cost normalization with adaptive exponent
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        Mean predictions from GP
    sigma : array of shape (2000,)
        Standard deviation predictions from GP
    f_best : float
        Best observed value so far
    cost : array of shape (2000,)
        Cost of evaluating each candidate
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores (higher = more desirable)
    """
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    # Ensure inputs are 1D arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = len(mu)
    
    # Safety: ensure finite outputs
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    cost = np.maximum(cost, 1e-6)  # Avoid division by zero
    
    # Adaptive parameters based on uncertainty level
    mean_sigma = np.mean(sigma)
    min_sigma = np.min(sigma)
    max_sigma = np.max(sigma)
    sigma_range = max_sigma - min_sigma
    
    # Explore more when uncertainty is high relative to best
    # This creates an adaptive balance between exploration and exploitation
    if f_best > 0:
        uncertainty_ratio = mean_sigma / (abs(f_best) + 1e-9)
    else:
        uncertainty_ratio = mean_sigma + 1e-9
    
    # Adaptive cost exponent: lower when exploring more (early), higher when exploiting (late)
    # Range: 0.4 (explore) to 1.2 (exploit)
    # When uncertainty is high relative to best, we want to explore more -> lower exponent
    cost_exponent = np.clip(0.4 + 0.8 * (1.0 - np.clip(uncertainty_ratio / 2.0, 0, 1)), 0.4, 1.2)
    
    # Expected Improvement
    z = (f_best - mu) / (sigma + 1e-9)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)  # EI should be non-negative
    
    # UCB for exploration
    # Beta parameter adapts: higher when uncertainty is high
    beta = 0.02 + 0.1 * np.clip(uncertainty_ratio, 0, 1)
    ucb = mu + beta * sigma
    
    # Combine EI and UCB
    # Weight shifts from UCB to EI as we get more confident
    # When sigma is high, favor UCB; when sigma is low, favor EI
    sigma_norm = sigma / (max_sigma + 1e-9)
    ei_weight = 1.0 - 0.3 * sigma_norm  # More EI when uncertainty is low
    ucb_weight = 0.3 * sigma_norm       # More UCB when uncertainty is high
    
    # Normalize ucb to be comparable to ei (both should be positive-ish)
    # Shift ucb so that its minimum is near 0
    ucb_shifted = ucb - np.min(ucb)
    ucb_shifted = np.maximum(ucb_shifted, 0.0)
    
    # Combine
    combined = ei_weight * ei + ucb_weight * ucb_shifted
    
    # Cost normalization with adaptive exponent
    # Lower cost exponent means less penalty for expensive evaluations
    cost_factor = cost ** cost_exponent
    scores = combined / cost_factor
    
    # Ensure finite and non-negative
    scores = np.where(np.isfinite(scores), scores, 0.0)
    scores = np.maximum(scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
