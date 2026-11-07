import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining Expected Improvement (EI)
    with an adaptive exploration bonus for multimodal landscapes.
    
    The core idea is to use EI per unit cost as the primary signal,
    but add a sigma-based exploration bonus scaled by the current
    best value to balance exploration and exploitation.
    
    Args:
        mu: (2000,) array of GP means.
        sigma: (2000,) array of GP standard deviations.
        f_best: float, the lowest observed value.
        cost: (2000,) array of costs for each candidate.
        
    Returns:
        (2000,) array of scores (higher is better).
    """
    t_start = time.time()
    deadline = 300.0
    
    # Ensure inputs are numpy arrays with proper dtype
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: ensure finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    
    # Ensure sigma and cost are positive
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Calculate z-score for Expected Improvement
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -10, 10)  # Clip to avoid numerical issues
    
    # Expected Improvement formula for minimization:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    ei = np.maximum(ei, 0.0)  # EI should be non-negative
    
    # Cost-aware EI: EI divided by cost
    cost_aware_ei = ei / cost_safe
    
    # Adaptive exploration bonus
    # When f_best is large (early in optimization), we want more exploration
    # When f_best is small (later in optimization), we want less exploration
    # Scale the bonus to be comparable to the EI term
    if f_best > 50.0:
        exploration_weight = 0.10
    elif f_best > 10.0:
        exploration_weight = 0.05
    elif f_best > 1.0:
        exploration_weight = 0.02
    else:
        exploration_weight = 0.01
    
    exploration_bonus = exploration_weight * sigma_safe / cost_safe
    
    # Combine: cost-aware EI + exploration bonus
    scores = cost_aware_ei + exploration_bonus
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check deadline
    t_elapsed = time.time() - t_start
    if t_elapsed > deadline - 5.0:
        # Safety margin: return current result
        return scores
    
    return scores
