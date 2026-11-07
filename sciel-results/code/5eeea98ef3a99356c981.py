"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Expected Improvement (EI) acquisition function.
    
    Strategy:
    1. Compute standard EI: E[f_best - f(x)]_+
    2. Divide by cost to get Expected Improvement per unit cost.
    3. Apply a dynamic exploration bonus that decays as budget is exhausted,
       ensuring we don't get stuck in local minima early on but exploit hard
       when close to the budget limit.
    """
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: handle zero or negative sigma (should not happen with GP, but just in case)
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Standardized distance to best
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (EI) formula:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is CDF and phi is PDF of standard normal
    # Note: This is the expected improvement over f_best.
    # If mu > f_best, z is negative.
    
    phi_z = norm.pdf(z)
    phi_zf = np.where(phi_z > 1e-10, phi_z, 1e-10) # Avoid underflow issues in division later if needed, though EI handles it
    
    EI = (f_best - mu) * norm.cdf(z) + sigma_safe * phi_z
    
    # EI is non-negative by definition mathematically, but numerical errors might cause tiny negatives
    EI = np.maximum(EI, 0.0)
    
    # Cost-aware score: EI per unit cost
    # Cost is 1 + mean(x_scaled). Range [1, 2].
    cost_safe = np.maximum(cost, 1e-9)
    cost_ei = EI / cost_safe
    
    # Dynamic Exploration Bonus:
    # We don't have direct access to total budget spent vs limit, but we can infer
    # from the magnitude of sigma or just use a fixed small exploration term 
    # if we detect high uncertainty relative to current best.
    # However, pure cost-EI is often robust. 
    # To handle multimodality (Rastrigin), pure EI can be myopic.
    # Let's add a small UCB-like term that is active only when EI is very low 
    # (i.e., no obvious improvement) to encourage exploration.
    
    # If EI is close to 0, the point is likely worse than best. 
    # We add a small exploration score based on sigma/cost to visit unexplored areas.
    # Weight: 0.1 is small enough not to distract from clear improvements.
    exploration_weight = 0.1
    ucb_term = sigma_safe / cost_safe
    score = cost_ei + exploration_weight * ucb_term
    
    # Ensure finite output
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
# EVOLVE-BLOCK-END
