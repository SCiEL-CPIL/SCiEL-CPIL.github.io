"""Cost-aware acquisition function for Bayesian optimization (powell 4d) - Cost-Aware Expected Improvement."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

_EPS = 1e-12
_DEADLINE_MARGIN = 5  # seconds to stop before the 330s wall-clock limit

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-Aware Expected Improvement acquisition function.
    
    Strategy:
    1. Compute the Expected Improvement (EI) for minimization: 
       EI = (f_best - mu) * Phi(z) + sigma * phi(z)
       where z = (f_best - mu) / sigma, Phi is the CDF of standard normal, 
       and phi is the PDF of standard normal.
    2. Scale the EI by the inverse of the evaluation cost to prioritize 
       cheap improvements. This is a standard approximation for cost-aware 
       Bayesian optimization (similar to Cost-Aware EI).
    3. Return the cost-aware EI scores.
    
    This approach is theoretically grounded and avoids the need for 
    arbitrary hyperparameter tuning (like kappa in UCB). It naturally 
    balances exploration (high sigma) and exploitation (low mu) based 
    on the expected gain per unit cost.
    """
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = mu.shape[0]
    if n == 0:
        return np.array([])

    # Deadline guard
    t0 = time.time()
    
    # Ensure sigma is positive for division
    sigma_safe = np.maximum(sigma, _EPS)
    
    # Compute z-score for improvement
    # For minimization, improvement is f_best - Y
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Compute EI: (f_best - mu) * Phi(z) + sigma * phi(z)
    # Phi(z) = norm.cdf(z)
    # phi(z) = norm.pdf(z)
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Note: (f_best - mu) can be negative, but EI should be non-negative.
    # The formula naturally handles this because if mu > f_best, z is negative,
    # Phi(z) is small, and the term becomes small. However, to be safe and 
    # strictly non-negative, we can clip the final EI to 0.
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # Ensure EI is non-negative (numerical stability)
    ei = np.maximum(ei, 0.0)
    
    # Check deadline
    if time.time() - t0 > _DEADLINE_MARGIN:
        # Fallback to simple cost-aware EI with no cost adjustment
        return ei

    # Cost-awareness: Divide by cost
    # The cost is 1 + mean(x_scaled). Lower cost is better.
    # We want to maximize EI / cost.
    cost_safe = np.maximum(cost, _EPS)
    
    # Cost-aware EI score
    score = ei / cost_safe
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Final deadline check
    if time.time() - t0 > _DEADLINE_MARGIN:
        return score
        
    return score

# EVOLVE-BLOCK-END
