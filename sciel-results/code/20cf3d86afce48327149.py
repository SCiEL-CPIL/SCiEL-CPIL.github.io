"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function blending Expected Improvement (EI) and UCB.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean predictions
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best observed value (lowest)
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next)
    """
    # Deadline guard: track elapsed time
    t0 = time.time()
    deadline = 280.0  # safety margin before 300s limit
    
    # Clip sigma to avoid division by zero and ensure positive values
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement (EI)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # UCB (Upper Confidence Bound) - we want to minimize, so use lower bound
    # In minimization, we prefer low mu and high sigma (uncertainty)
    # UCB for minimization: -mu + beta * sigma
    beta = 2.0  # exploration-exploitation balance
    ucb = -mu + beta * sigma_safe
    
    # Normalize EI and UCB to comparable scales
    ei_max = np.max(ei)
    if ei_max > 1e-12:
        ei_norm = ei / ei_max
    else:
        ei_norm = ei
    
    ucb_min = np.min(ucb)
    ucb_max = np.max(ucb)
    ucb_range = ucb_max - ucb_min
    if ucb_range > 1e-12:
        ucb_norm = (ucb - ucb_min) / ucb_range
    else:
        ucb_norm = ucb
    
    # Blend EI and UCB
    # Weight more towards EI as we progress (but we don't have explicit progress info)
    # Use a fixed blend: 70% EI, 30% UCB
    blend_weight_ei = 0.7
    blended = blend_weight_ei * ei_norm + (1 - blend_weight_ei) * ucb_norm
    
    # Cost-aware: divide by cost to get improvement per unit cost
    # Ensure cost is positive
    cost_safe = np.maximum(cost, 1e-9)
    scores = blended / cost_safe
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check deadline
    if time.time() - t0 > deadline:
        # Return best result found so far
        return scores
    
    return scores
# EVOLVE-BLOCK-END
