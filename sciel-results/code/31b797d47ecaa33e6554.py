"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Combines Expected Improvement (EI) with cost-awareness and adds
    an exploration bonus based on sigma to balance exploitation and exploration.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean predictions from the GP
    sigma : ndarray, shape (2000,)
        Standard deviation predictions from the GP
    f_best : float
        Best (lowest) observed value so far
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate point
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next)
    """
    # Ensure we have finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Standardize for numerical stability in z-score computation
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid extreme values in norm.cdf and norm.pdf
    z = np.clip(z, -10, 10)
    
    # Expected Improvement (EI)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware: divide by cost to get EI per unit cost
    cost_safe = np.maximum(cost, 1e-8)
    cost_aware_ei = ei / cost_safe
    
    # Exploration bonus: encourage evaluating points with high uncertainty
    # Use a weighted combination that balances exploitation and exploration
    # The exploration term is scaled to be comparable to the exploitation term
    max_ei = np.max(ei) if len(ei) > 0 else 1.0
    max_cost_aware = np.max(cost_aware_ei) if len(cost_aware_ei) > 0 else 1.0
    
    # Scale exploration bonus relative to the scale of exploitation
    exploration_scale = max_cost_aware * 0.1  # 10% weight for exploration
    exploration_bonus = sigma_safe * exploration_scale / cost_safe
    
    # Combined score
    scores = cost_aware_ei + exploration_bonus
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
