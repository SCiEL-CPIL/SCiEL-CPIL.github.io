"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 320.0  # 10s safety margin before 330s kill


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Linear Lower Confidence Bound (LBC)
    with cost normalization.
    
    score = (f_best - mu + beta * sigma) / (cost + eps)
    
    Parameters:
    -----------
    mu : array-like, shape (n,)
        GP mean predictions
    sigma : array-like, shape (n,)
        GP standard deviation predictions
    f_best : float
        Best (lowest) observed value so far
    cost : array-like, shape (n,)
        Cost of evaluating each candidate point
    
    Returns:
    --------
    scores : ndarray, shape (n,)
        Acquisition values (higher = more desirable to evaluate)
    """
    # Deadline guard: return simple fallback to avoid timeout
    if time.time() > _DEADLINE:
        mu = np.asarray(mu, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        cost = np.asarray(cost, dtype=np.float64)
        sigma_safe = np.maximum(sigma, 1e-9)
        score = (f_best - mu + 1.5 * sigma_safe) / (cost + 1e-9)
        return np.where(np.isfinite(score), score, 0.0)

    # Convert to numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Ensure sigma is non-negative
    sigma_safe = np.maximum(sigma, 1e-9)

    # Expected Improvement for minimization:
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    # where Phi and phi are the standard normal CDF and PDF
    from scipy.stats import norm
    
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost-aware scoring: normalize EI by cost
    # Also add a small exploration bonus weighted by sigma to encourage diversity
    # The sigma term helps explore regions of high uncertainty
    cost_norm = cost / (cost.mean() + 1e-9)  # normalize cost to mean 1
    
    # Hybrid: primarily EI/cost, with a small sigma-based exploration bonus
    # This encourages both finding improvements and exploring uncertain regions
    exploration_weight = 0.3
    score = ei / (cost_norm + 1e-9) + exploration_weight * sigma_safe / (cost_norm + 1e-9)
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)

    return score
# EVOLVE-BLOCK-END
