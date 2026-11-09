# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Combines cost-normalized Expected Improvement (EI) with an uncertainty
    bonus, and a cost-penalty term to balance exploitation, exploration,
    and evaluation cost.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Mean predictions from GP.
    sigma : array-like, shape (2000,)
        Standard deviation predictions from GP.
    f_best : float
        Best (lowest) observed function value.
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = more desirable to evaluate).
    """
    # Deadline guard
    t_start = time.time()
    
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Compute z-score for expected improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost-normalized EI (exploitation component)
    ei_normalized = ei / cost_safe
    
    # Exploration bonus: encourages visiting uncertain regions
    # Scaled by 1/cost to maintain cost-awareness
    exploration_bonus = 0.08 * sigma_safe / cost_safe
    
    # Cost penalty: discourages expensive evaluations when EI is low
    # This allows high-EI points in expensive regions if improvement outweighs cost
    cost_penalty = 0.02 * cost_safe
    
    # Combine: exploitation + exploration - cost penalty
    scores = ei_normalized + exploration_bonus - cost_penalty
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, -1e6)
    
    # Check deadline
    elapsed = time.time() - t_start
    if elapsed > 280:  # Safety margin before 300s limit
        return scores
    
    return scores
# EVOLVE-BLOCK-END
