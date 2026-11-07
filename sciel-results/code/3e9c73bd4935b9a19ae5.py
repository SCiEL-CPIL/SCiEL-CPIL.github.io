"""Cost-aware acquisition function for Bayesian optimization (shekel 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import erf


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for minimizing shekel 4d.
    
    Uses a Lower Confidence Bound (LCB) with a cost-scaled exploration bonus.
    The LCB explicitly trades off exploitation (low mu) against exploration (high sigma),
    and the exploration bonus is inversely scaled by cost to favor cheap, uncertain regions.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean predictions (in objective units).
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions (in objective units).
    f_best : float
        Lowest observed function value.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    t0 = time.time()
    
    # Safety deadline guard: leave margin before 330s wall-clock
    DEADLINE = 320.0  # seconds
    
    # Handle edge cases
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-9)
    
    # --- Lower Confidence Bound (LCB) for minimization ---
    # LCB(x) = mu(x) - kappa * sigma(x)
    # We want to minimize LCB, so we maximize -LCB(x) = -(mu - kappa * sigma)
    # 
    # Key structural choice: the exploration bonus kappa is dynamically scaled
    # by the inverse of cost, so uncertain regions with low cost are prioritized.
    # This creates a cost-aware exploration signal that naturally balances
    # exploitation vs. exploration based on the cost landscape.
    
    # Base exploration parameter
    kappa_base = 1.0
    
    # Dynamic scaling: exploration bonus is inversely proportional to cost.
    # This means:
    # - Cheap points (low cost) get a larger effective exploration bonus
    # - Expensive points (high cost) get a smaller effective exploration bonus
    # This naturally encourages exploring cheap, uncertain regions.
    
    # Direct cost normalization: divide LCB by cost to create strong cost-awareness
    # Cost ranges from 1.0 to 2.0, so dividing by cost gives scores in [0.5, 1.0] range
    # This strongly favors cheap, uncertain regions over expensive, uncertain regions
    
    # Compute LCB: mu - kappa * sigma (with kappa_base = 1.0)
    lcb = mu - kappa_base * sigma_safe
    
    # Acquisition score: -(LCB) / cost
    # Lower LCB (more promising) and lower cost both increase the score
    scores = -lcb / cost_safe
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Deadline check
    if time.time() - t0 > DEADLINE:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
