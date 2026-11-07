import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-normalized Expected Improvement (EI / cost) acquisition function.
    
    Strategy: Use Expected Improvement for minimization and divide by the
    evaluation cost. This directly optimizes expected improvement per unit
    cost, aligning the search with the budget-constrained objective without
    arbitrary additive or multiplicative bias parameters.
    
    The EI component naturally balances exploration and exploitation as the
    GP posterior updates. Dividing by cost (which ranges from 1 to 2)
    inherently prefers cheaper evaluations when predicted improvements are
    similar, conserving the total budget of 30 units.
    """
    # Deadline guard: if we're close to the time limit, return a simple fallback
    elapsed = time.time() - _start_time
    if elapsed > 280:
        # Fallback: simple greedy exploitation with cost normalization
        sigma_safe = np.maximum(sigma, 1e-9)
        scores = (-mu) / np.maximum(cost, 1e-9)
        return np.where(np.isfinite(scores), scores, 0.0)
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score for EI (minimization)
    # z = (f_best - mu) / sigma
    diff = f_best - mu
    z = diff / sigma_safe
    
    # Clip z to prevent numerical issues in CDF/PDF
    z_clipped = np.clip(z, -30, 30)
    
    # Calculate Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    phi_z = norm.pdf(z_clipped)
    Phi_z = norm.cdf(z_clipped)
    
    ei = diff * Phi_z + sigma_safe * phi_z
    
    # Ensure EI is non-negative (numerical artifacts might make it slightly negative)
    ei = np.maximum(ei, 0.0)
    
    # Cost normalization: divide by cost to get EI per unit cost
    # cost is in [1, 2], so this scales the EI down for more expensive points
    cost_safe = np.maximum(cost, 1e-9)
    scores = ei / cost_safe
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
