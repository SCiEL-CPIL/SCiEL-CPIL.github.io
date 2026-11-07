import numpy as np
from scipy.stats import norm
import time

# Cost-aware acquisition function for Bayesian optimization (levy 2d)
# We use a cost-aware expected improvement with an adaptive exploration bonus.
# The function is called with mu, sigma, f_best, cost all of shape (2000,).

# Track call count to adapt exploration
_call_count = 0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function: Expected Improvement per unit cost
    with an adaptive exploration bonus to balance exploitation and exploration.
    """
    global _call_count
    _call_count += 1
    
    # Compute z-score
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    ei = (f_best - mu) * Phi_z + sigma * phi_z
    
    # Adaptive exploration parameter: start with more exploration, decay over time
    # kappa controls the exploration-exploitation tradeoff
    # Early on, we want more exploration; later, more exploitation
    kappa = max(0.1, 1.0 / (1.0 + 0.05 * _call_count))
    
    # Add exploration bonus based on uncertainty
    # This helps escape local minima in the acquisition landscape
    exploration_bonus = kappa * sigma_safe
    
    # Cost-aware: divide by cost, add exploration bonus
    cost_safe = np.maximum(cost, 1e-9)
    score = ei / cost_safe + exploration_bonus / cost_safe
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
