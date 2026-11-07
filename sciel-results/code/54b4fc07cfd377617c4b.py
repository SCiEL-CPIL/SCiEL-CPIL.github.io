"""Rewritten acquisition function using cost-aware EI with dynamic exploration bonus."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Track elapsed time for deadline guard
_start_time = time.time()
_DEADLINE = 300.0  # seconds, safety margin before 330s kill


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses a combination of Expected Improvement (EI) normalized by cost,
    with an exploration bonus for high-uncertainty, low-cost points.
    
    Parameters
    ----------
    mu : array_like, shape (2000,)
        GP mean predictions in objective units.
    sigma : array_like, shape (2000,)
        GP standard deviation predictions in objective units.
    f_best : float
        Best observed objective value so far.
    cost : array_like, shape (2000,)
        Cost of evaluating each candidate.
    
    Returns
    -------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = more desirable to evaluate next).
    """
    # Check deadline
    if time.time() - _start_time > _DEADLINE:
        # Fallback: return uniform scores or zero scores if we are too late
        # But we should try to compute quickly if possible, or just return safe values
        # Since we are inside the loop, we must return *something* valid.
        # Returning zeros is safe but useless. Returning 1/cost is a reasonable fallback.
        cost_arr = np.asarray(cost, dtype=np.float64)
        cost_arr = np.where(np.isfinite(cost_arr) & (cost_arr > 0), cost_arr, 1.0)
        return 1.0 / cost_arr

    # Ensure arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: replace non-finite values
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    
    # Calculate Expected Improvement (EI) for minimization
    # EI = E[max(f_best - y, 0)] where y ~ N(mu, sigma^2)
    # Z = (f_best - mu) / sigma
    # EI = (f_best - mu) * Phi(Z) + sigma * phi(Z)
    
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    z = (f_best - mu) / sigma_safe
    
    # Compute Phi(Z) and phi(Z)
    phi_z = norm.pdf(z)
    phi_cdf_z = norm.cdf(z)
    
    # EI calculation
    ei = (f_best - mu) * phi_cdf_z + sigma_safe * phi_z
    
    # Numerical stability: EI should be non-negative
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score: EI per unit cost
    base_score = ei / cost
    
    # Exploration bonus: encourages visiting high-uncertainty, low-cost points
    # This helps in early optimization or flat regions where EI might be low
    # but uncertainty is high.
    exploration_bonus = 0.05 * sigma / cost
    
    # Combine scores
    scores = base_score + exploration_bonus
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
