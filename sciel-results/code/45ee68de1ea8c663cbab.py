"""Cost-aware acquisition function for Bayesian optimization (cosine8 8d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = 330.0
_SAFETY_MARGIN = 5.0


def _median_mad_norm(arr):
    """
    Robust normalization using Median and Median Absolute Deviation (MAD).
    This is more robust to outliers than mean/std z-scoring, which is
    critical in high-dimensional spaces where candidate scores can be skewed.
    """
    med = np.median(arr)
    mad = np.median(np.abs(arr - med))
    
    # If MAD is very small, the distribution is likely constant or near-constant.
    # Fall back to standard z-score if possible, else zeros.
    if mad < 1e-8:
        mean = np.mean(arr)
        std = np.std(arr)
        if std < 1e-8:
            return np.zeros_like(arr)
        return (arr - mean) / std
        
    # 1.4826 is the consistency constant for the MAD estimator of sigma for normal distributions.
    # Using it makes the scale comparable to standard deviation.
    return (arr - med) / (1.4826 * mad)


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition using robustly normalized Expected Improvement (EI)
    and Upper Confidence Bound (UCB).
    
    Parameters:
    -----------
    mu : np.ndarray
        Mean of the GP posterior, shape (2000,).
    sigma : np.ndarray
        Standard deviation of the GP posterior, shape (2000,).
    f_best : float
        Best observed value so far.
    cost : np.ndarray
        Cost of evaluating each candidate, shape (2000,).
        
    Returns:
    --------
    np.ndarray
        Acquisition scores, shape (2000,). Higher is better.
    """
    # Deadline guard
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE - _SAFETY_MARGIN:
        return np.zeros(len(mu))
    
    # Ensure positive values to avoid division by zero or negative costs
    sigma = np.maximum(sigma, 1e-8)
    cost = np.maximum(cost, 1e-8)
    
    # 1. Expected Improvement (EI)
    # Standard EI for minimization: E[max(f_best - f, 0)]
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # 2. Cost-aware EI
    # This term represents the expected improvement per unit cost.
    ei_cost = ei / cost
    
    # 3. Upper Confidence Bound (UCB) for exploration
    # We use a moderate beta. Since we normalize robustly, we don't need to
    # tune beta as aggressively to balance exploration vs exploitation.
    beta = 1.5
    ucb = -mu + beta * sigma
    ucb_cost = ucb / cost
    
    # 4. Diversity/Uncertainty term
    # Encourages exploring high-uncertainty regions, penalized by cost.
    div_cost = sigma / cost
    
    # 5. Robust Normalization
    # Using Median/MAD normalization to handle potential skewness in the
    # distribution of the 2000 candidates.
    z_ei = _median_mad_norm(ei_cost)
    z_ucb = _median_mad_norm(ucb_cost)
    z_div = _median_mad_norm(div_cost)
    
    # 6. Combine components
    # Weights: 
    # - 0.70 for Exploitation (EI)
    # - 0.25 for Exploration (UCB)
    # - 0.05 for pure Uncertainty (Diversity)
    # The robust normalization allows us to rely more on EI as the signal-to-noise ratio is better.
    scores = 0.70 * z_ei + 0.25 * z_ucb + 0.05 * z_div
    
    # 7. Ensure finite
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
