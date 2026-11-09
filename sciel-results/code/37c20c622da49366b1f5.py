"""Cost-aware acquisition for Bayesian optimization (styblinski tang 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Module-level deadline guard
_DEADLINE = None
_TIME_LIMIT = 325.0  # seconds, safety margin before 330s kill

def _get_deadline():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + _TIME_LIMIT
    return _DEADLINE

def _time_remaining():
    return max(0.0, _get_deadline() - time.time())


def acquisition(mu, sigma, f_best, cost):
    """
    Return 2000 finite scores (higher = evaluate next).
    
    Strategy: Use a cost-aware acquisition that combines Expected Improvement
    normalized by the mean cost (hypervolume-style global efficiency metric)
    with an uncertainty bonus. This creates a dynamic resource allocation
    perspective where the threshold for accepting expensive points adapts
    to the overall cost structure of the candidate set.
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: ensure no NaNs or infs in inputs
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=1e-9, posinf=1e-9, neginf=-1e-9)
    cost = np.nan_to_num(cost, nan=1.0, posinf=1.0, neginf=1.0)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-6)
    
    # Compute Expected Improvement (EI)
    z = (f_best - mu) / sigma
    z = np.clip(z, -10, 10)
    
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Hypervolume-based normalization: use mean cost as a global efficiency metric
    # This normalizes EI by the average cost of the candidate set, creating
    # a resource-allocation perspective that stabilizes exploration-exploitation
    mean_cost = np.mean(cost)
    mean_cost = max(mean_cost, 1e-6)
    
    # EI normalized by mean cost (global efficiency)
    ei_global = ei / mean_cost
    
    # Also compute EI per individual cost for local efficiency
    ei_local = ei / cost
    
    # Combine global and local efficiency with a weighted sum
    # Weight toward global to create trajectory-level resource allocation
    w_global = 0.7
    w_local = 0.3
    ei_combined = w_global * ei_global + w_local * ei_local
    
    # Uncertainty bonus: sigma normalized by mean cost
    ucb_global = sigma / mean_cost
    
    # Uncertainty bonus: sigma per individual cost
    ucb_local = sigma / cost
    
    # Combine UCB terms
    alpha_global = 0.4
    alpha_local = 0.2
    ucb_combined = alpha_global * ucb_global + alpha_local * ucb_local
    
    # Final score
    scores = ei_combined + ucb_combined
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
