"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

# Deadline guard: must return well before the 330s wall-clock kill.
_DEADLINE = time.time() + 300.0  # 300s margin
_DEADLINE_HARD = time.time() + 325.0


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition for Bayesian optimization on rosenbrock 2d.
    
    Parameters
    ----------
    mu : array of shape (2000,) - GP mean predictions
    sigma : array of shape (2000,) - GP std dev predictions
    f_best : float - lowest observed value
    cost : array of shape (2000,) - evaluation costs
    
    Returns
    -------
    scores : array of shape (2000,) - acquisition scores (higher = evaluate next)
    """
    start = time.time()
    
    # Basic safety: ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Compute z for EI
    diff = f_best - mu
    z = diff / sigma_safe
    z = np.clip(z, -50, 50)
    
    # Expected Improvement (EI)
    ei = diff * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Structurally different approach: Use a combination of:
    # 1. Cost-aware EI (exploitation)
    # 2. Inverse variance weighted exploration (for uncertain regions)
    # 3. Adaptive cost penalty based on expected improvement
    
    # Cost-aware EI
    ei_cost_aware = ei / cost_safe
    
    # Exploration term: favor high uncertainty regions, but penalize by cost
    # Use inverse variance to weight exploration - high sigma means high uncertainty
    # But we want to balance this with exploitation
    exploration = sigma_safe / (1.0 + sigma_safe)  # Bounded exploration score
    exploration_cost_aware = exploration / cost_safe
    
    # Adaptive weighting: use more exploration when EI signal is weak
    # Normalize EI to get a relative measure of exploitation strength
    ei_max = np.max(ei_cost_aware) if np.max(ei_cost_aware) > 0 else 1.0
    ei_normalized = ei_cost_aware / (ei_max + 1e-9)
    
    # When EI is low (relative to best), increase exploration weight
    # This helps escape local minima
    exploration_weight = 0.1 + 0.3 * (1.0 - ei_normalized)
    exploitation_weight = 1.0 - exploration_weight
    
    # Combine
    score = exploitation_weight * ei_cost_aware + exploration_weight * exploration_cost_aware
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, -np.inf)
    
    # Check deadline
    if time.time() > _DEADLINE:
        return score
    
    return score
# EVOLVE-BLOCK-END
