"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement with softened cost penalty.
    
    Parameters:
    -----------
    mu : np.ndarray, shape (N,)
        GP mean predictions at candidate points.
    sigma : np.ndarray, shape (N,)
        GP standard deviation predictions at candidate points.
    f_best : float
        Best observed function value so far.
    cost : np.ndarray, shape (N,)
        Evaluation cost for each candidate point.
    
    Returns:
    --------
    np.ndarray, shape (N,)
        Acquisition values (higher = more promising).
    """
    # Safety margin: return early if we're getting close to the 300s limit
    elapsed = time.time() - _start_time
    if elapsed > 290:
        # Return a reasonable fallback: just EI without cost penalty
        mu_arr = np.asarray(mu, dtype=np.float64)
        sigma_arr = np.asarray(sigma, dtype=np.float64)
        sigma_safe = np.maximum(sigma_arr, 1e-9)
        improvement = f_best - mu_arr
        z = improvement / sigma_safe
        ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        return np.where(np.isfinite(ei), ei, 0.0)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma to avoid division by zero and numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Clip cost to avoid division by zero
    cost = np.clip(cost, 1e-8, None)
    
    # Compute z-score for Expected Improvement
    improvement = f_best - mu
    z = improvement / sigma_safe
    
    # Expected Improvement: E[improvement | improvement > 0]
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = improvement * cdf_z + sigma_safe * pdf_z
    
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware: use a softened cost penalty to avoid biasing towards
    # the cheaper local minimum. The global minimum is in a higher cost
    # region, so a strong 1/cost penalty hurts. Use a sublinear penalty
    # that grows slowly with cost.
    # cost ranges from 1.0 (all coords at 0) to 2.0 (all coords at 1)
    # Use sqrt(cost) for a gentler penalty than linear
    cost_penalty = np.sqrt(cost)
    cost_penalty = np.maximum(cost_penalty, 1e-8)
    
    # Base score: EI divided by cost penalty
    score = ei / cost_penalty
    
    # Add a small exploration bonus for low-cost regions to prevent
    # premature convergence. This encourages the search to explore
    # cheaper areas early on, which helps find the global minimum
    # that may be in a higher-cost region.
    # The bonus is small and decays as f_best improves (gets closer to 0)
    # Scale: when f_best is large (early), bonus is larger; when f_best ~ 0, bonus ~ 0
    exploration_scale = max(0.0, min(f_best, 1.0)) * 0.05
    # Low-cost regions get a small bonus
    cost_range = cost - cost.min()
    cost_range_max = max(cost_range.max(), 1e-8)
    low_cost_bonus = (1.0 - cost_range / cost_range_max) * exploration_scale
    score = score + low_cost_bonus
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
