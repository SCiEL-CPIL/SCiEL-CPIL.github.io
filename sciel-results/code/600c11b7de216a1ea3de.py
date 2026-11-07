"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global timer for deadline guard
_START_TIME = time.time()
_DEADLINE = 330.0
_SAFETY_MARGIN = 5.0


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function with exploration bonus.
    
    Args:
        mu: Array of mean predictions (2000,).
        sigma: Array of standard deviations (2000,).
        f_best: Scalar, best observed value.
        cost: Array of evaluation costs (2000,).
        
    Returns:
        Array of scores (2000,), higher is better.
    """
    # Deadline guard
    if time.time() - _START_TIME > _DEADLINE - _SAFETY_MARGIN:
        # Fallback to simple EI/cost to ensure finite output quickly
        z = (f_best - mu) / (sigma + 1e-9)
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        return ei / cost

    # Calculate Z-score
    z = (f_best - mu) / (sigma + 1e-9)
    
    # Calculate Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    phi_z = norm.pdf(z)
    phi_cdf_z = norm.cdf(z)
    ei = (f_best - mu) * phi_cdf_z + sigma * phi_z
    
    # Add a small exploration bonus to encourage moving away from local minima
    # This is a structural change from pure EI
    exploration_bonus = 0.1 * sigma
    
    # Total gain
    total_gain = ei + exploration_bonus
    
    # Apply cost discount
    # Cost is in [1, 2]
    scores = total_gain / cost
    
    # Ensure finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
