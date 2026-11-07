"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses an adaptive UCB (Upper Confidence Bound) strategy normalized by cost.
    The UCB balances exploration (via kappa*sigma) and exploitation (via mu),
    with kappa adapting based on how close f_best is to the known minimum (0).
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean predictions from GP.
    sigma : ndarray, shape (2000,)
        Standard deviation predictions from GP.
    f_best : float
        Best observed function value so far.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = more desirable to evaluate next).
    """
    # Track elapsed time for deadline guard
    t_start = time.time()
    deadline = t_start + 290  # Safety margin before 300s limit
    
    # Ensure sigma is positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Adaptive kappa: start with moderate exploration, reduce as we converge
    # When f_best is close to 0 (the minimum), we want to exploit more
    # Use a smooth transition: kappa = base_kappa * (1 + f_best / scale)
    # But clamp to reasonable range
    base_kappa = 1.5
    # As f_best -> 0, kappa should decrease to focus on exploitation
    # f_best is non-negative for three hump camel (min is 0)
    # Use: kappa = base_kappa * (0.3 + 0.7 * min(f_best / 1.0, 1.0))
    # This gives kappa between 0.45 and 1.5
    f_best_clamped = min(max(f_best, 0.0), 10.0)
    kappa = base_kappa * (0.3 + 0.7 * (f_best_clamped / 1.0))
    kappa = min(max(kappa, 0.3), 2.0)
    
    # UCB: mu - kappa * sigma (for minimization, we want low mu and high uncertainty)
    # Wait - for minimization, we want to minimize f. UCB for minimization:
    # We want points with low predicted mean AND high uncertainty (could be even lower)
    # So the acquisition should be: -mu + kappa * sigma (higher is better for selection)
    # Or equivalently, we minimize mu - kappa*sigma, but we want higher scores to be selected
    # So: score = (-mu + kappa * sigma) / cost
    
    # Actually, let me reconsider. The task says "higher = evaluate next".
    # For minimization:
    # - Low mu is good (predicted low value)
    # - High sigma is good (uncertainty, could be even lower)
    # So we want to maximize: -mu + kappa * sigma
    # Then normalize by cost.
    
    ucb = -mu + kappa * sigma_safe
    
    # Normalize by cost (cost is positive, 1 + mean(x_scaled))
    # Higher score = better
    scores = ucb / np.maximum(cost, 1e-10)
    
    # Check deadline
    if time.time() > deadline:
        # Return what we have - should be fast enough normally
        pass
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
