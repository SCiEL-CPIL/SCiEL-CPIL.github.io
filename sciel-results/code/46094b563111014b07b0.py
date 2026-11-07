"""Cost-aware acquisition function for Bayesian optimization (rastrigin 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 295  # 295 seconds limit (5s safety margin before 300s kill)

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-Effective Expected Improvement (CEI) for Bayesian optimization.
    
    Args:
        mu: array of shape (2000,) - GP mean
        sigma: array of shape (2000,) - GP standard deviation
        f_best: float - best observed value
        cost: array of shape (2000,) - cost of evaluating each point
        
    Returns:
        scores: array of shape (2000,) - acquisition scores (higher is better)
    """
    # Check deadline
    if time.time() > _DEADLINE:
        # Fallback: return 1/cost to prefer cheap points
        return 1.0 / (np.maximum(cost, 1e-9))
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score for Expected Improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (for minimization: improvement = f_best - f(x))
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # CEI: Expected Improvement divided by cost
    # This naturally balances finding improvements against evaluation cost
    cei = ei / np.maximum(cost, 1e-9)
    
    # Add adaptive exploration component
    # When f_best is still high (early in search), explore more
    # When f_best is low (later in search), exploit more
    # Use a sigmoid-like transition based on f_best
    # Rastrigin min is 0, so f_best starts around 20-30 and decreases
    exploration_weight = 1.0 / (1.0 + np.exp((f_best - 5.0) / 2.0))
    
    # Exploration bonus: prefer high uncertainty points
    # Scale by the typical uncertainty level
    sigma_mean = np.mean(sigma_safe)
    if sigma_mean > 1e-6:
        exploration_bonus = exploration_weight * 0.3 * sigma_safe / sigma_mean
    else:
        exploration_bonus = 0.0
    
    # Combine CEI with exploration
    final_scores = cei + exploration_bonus
    
    # Ensure all scores are finite
    final_scores = np.where(np.isfinite(final_scores), final_scores, 0.0)
    final_scores = np.maximum(final_scores, 0.0)
    
    return final_scores
# EVOLVE-BLOCK-END
