"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global start time to track deadline
_START_TIME = time.time()
_DEADLINE = _START_TIME + 300.0  # 300s safety margin within 330s limit

def _time_remaining():
    return _DEADLINE - time.time()

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Expected Improvement (EI) acquisition function.
    
    Strategy:
    Uses the standard Expected Improvement formula, which is known to be highly
    effective for simple regret in Bayesian Optimization. The key difference from
    the previous UCB approach is that EI directly optimizes the expected reduction
    in the best observed value, weighted by the inverse of the evaluation cost.
    
    EI(x) = E[max(f_best - f(x), 0)]
    Cost-aware score = EI(x) / cost(x)
    
    Parameters:
    mu: (2000,) array of GP means
    sigma: (2000,) array of GP standard deviations
    f_best: scalar, best observed value (lowest)
    cost: (2000,) array of evaluation costs
    
    Returns:
    scores: (2000,) array of acquisition scores (higher is better)
    """
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety check for time
    if _time_remaining() < 1.0:
        # If running out of time, return a simple heuristic quickly
        # Just return mu/cost to avoid complex computations
        safe_cost = np.maximum(cost, 1e-9)
        return mu / safe_cost

    # Avoid division by zero and negative sigma
    safe_cost = np.maximum(cost, 1e-6)
    safe_sigma = np.maximum(sigma, 1e-6)
    
    # Calculate Expected Improvement
    # z = (f_best - mu) / sigma
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is CDF and phi is PDF of standard normal
    
    diff = f_best - mu
    z = diff / safe_sigma
    
    # Clip z to avoid numerical issues in extreme tails
    z = np.clip(z, -50, 50)
    
    ei = diff * norm.cdf(z) + safe_sigma * norm.pdf(z)
    
    # Cost-aware normalization
    # We want to maximize EI per unit cost
    scores = ei / safe_cost
    
    # Ensure finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
