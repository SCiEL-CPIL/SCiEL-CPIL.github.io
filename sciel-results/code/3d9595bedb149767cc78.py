# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Deadline guard: 330s wall-clock limit, use 325s as safety margin
_DEADLINE = 325.0
_START_TIME = time.time()

def _check_deadline():
    """Return True if we should abort and return current best."""
    return (time.time() - _START_TIME) > _DEADLINE

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining Expected Improvement and
    a cost-adjusted exploration bonus.
    
    Strategy:
    - Compute standard Expected Improvement (EI) for exploitation.
    - Add a cost-adjusted exploration term that rewards high uncertainty
      at low cost, helping escape local optima in multi-modal landscapes.
    - The cost penalty is applied to the EI term to discourage expensive
      evaluations when the expected gain is modest.
    """
    # Check deadline first
    if _check_deadline():
        # Return a neutral score if we're running out of time
        return np.zeros_like(mu, dtype=np.float64)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    
    # Clip sigma to avoid numerical issues in z calculation
    sigma = np.clip(sigma, 1e-6, 1e6)
    cost = np.clip(cost, 0.5, 100.0)
    
    # Calculate Expected Improvement
    z = (f_best - mu) / sigma
    
    cdf_vals = norm.cdf(z)
    pdf_vals = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_vals + sigma * pdf_vals
    
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0.0)
    
    # Cost-adjusted EI: divide by cost to penalize expensive evaluations
    # This makes cheap points with moderate EI compete better with
    # expensive points with high EI
    ei_cost_adjusted = ei / cost
    
    # Exploration bonus: reward high uncertainty at low cost
    # The term sigma / cost encourages exploring uncertain regions
    # but only when they are cheap to evaluate
    # Scale factor of 0.5 balances exploration vs exploitation
    exploration_bonus = 0.5 * sigma / cost
    
    # Combine: exploitation (cost-adjusted EI) + exploration (uncertainty/cost)
    score = ei_cost_adjusted + exploration_bonus
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
