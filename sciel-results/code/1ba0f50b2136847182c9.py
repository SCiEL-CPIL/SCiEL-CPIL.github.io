import numpy as np
from scipy.stats import norm
import time

# Module-level state for time tracking
_start_time = None

def _get_elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Powell 4D BO.
    
    Parameters:
    - mu: (2000,) array of GP mean predictions
    - sigma: (2000,) array of GP std predictions  
    - f_best: scalar, best observed value so far
    - cost: (2000,) array of evaluation costs for each candidate
    
    Returns:
    - (2000,) array of acquisition scores (higher = better)
    """
    # Time guard: if we've been running too long, return safe fallback
    elapsed = _get_elapsed()
    if elapsed > 280:  # Safety margin before 300s limit
        # Return simple EI/cost as fallback
        z = (f_best - mu) / (sigma + 1e-9)
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        return ei / (cost + 1e-9)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safe sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Z-score for Probability of Improvement
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement: P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    pi = norm.cdf(z)
    
    # Expected Improvement for minimization
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Cost-aware weighting:
    # Use a soft cost penalty via exponentiation on PI
    # PI^(1/cost) naturally balances exploration and cost
    # When cost is low (cheap), exponent is high, requiring high PI
    # When cost is high (expensive), exponent is low, allowing moderate PI
    
    cost_safe = np.maximum(cost, 1e-6)
    
    # Combine PI with cost-aware exponentiation
    pi_cost_aware = np.power(np.maximum(pi, 1e-15), 1.0 / cost_safe)
    
    # Add exploration term to encourage visiting uncertain regions
    # This helps escape local minima and cover the space more effectively
    # Use a moderate exploration coefficient that scales with uncertainty
    exploration_lambda = 0.5
    exploration_term = exploration_lambda * sigma / cost_safe
    
    # Combine exploitation (PI-based) and exploration (sigma-based)
    # Weight the PI term more heavily since it's the primary signal
    score = pi_cost_aware + exploration_term
    
    # Ensure finite values
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
