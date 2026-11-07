"""Acquisition function for Bayesian optimization of styblinski tang 1d.
Uses Adaptive GP-UCB with a dynamic exploration coefficient and an 
exploitation bias to balance global search and local refinement 
within the 30-evaluation budget.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Module-level deadline guard
_start_time = time.time()
_DEADLINE = _start_time + 300.0  # 300s wall-clock limit per evaluation


def acquisition(mu, sigma, f_best):
    """
    Compute acquisition scores for 2000 candidate points.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        GP posterior mean at candidates (in objective units)
    sigma : array of shape (2000,)
        GP posterior standard deviation at candidates (in objective units)
    f_best : float
        Best (lowest) observed value so far
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores; higher means evaluate next
    """
    # Deadline safety check
    if time.time() > _DEADLINE - 5.0:
        # Fallback: Simple UCB with moderate exploration
        sigma_safe = np.maximum(sigma, 1e-9)
        return -mu + 2.0 * sigma_safe

    # Clip sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Define reference values for the Styblinski-Tang function on [-5, 5]
    # Minimum is approx -39.166, maximum is approx 40 (at x=5 or x=-5)
    f_min_ref = -39.166
    f_max_ref = 40.0
    
    # Calculate progress: 0.0 at start (f_best ~ 40), 1.0 at end (f_best ~ -39)
    # If f_best is worse than f_max_ref, progress is 0.
    # If f_best is better than f_min_ref, progress is 1.
    progress = (f_max_ref - f_best) / (f_max_ref - f_min_ref)
    progress = np.clip(progress, 0.0, 1.0)
    
    # Dynamic exploration coefficient (kappa)
    # Start with high exploration to find the global minimum basin.
    # Decay to low exploration to refine the solution.
    # Using a square root decay allows for more exploration in the mid-game.
    kappa_start = 8.0
    kappa_end = 0.5
    kappa = kappa_end + (kappa_start - kappa_end) * (1.0 - progress) ** 0.5
    
    # Base UCB score for minimization: -mu + kappa * sigma
    # This encourages low mu (exploitation) and high sigma (exploration)
    scores = -mu + kappa * sigma_safe
    
    # Add an exploitation bias: 
    # If the GP predicts a value significantly lower than f_best, 
    # boost the score to ensure we converge on promising regions 
    # even if uncertainty drops.
    # This helps in the later stages where sigma is small but mu is accurate.
    improvement_potential = np.maximum(f_best - mu, 0.0)
    # Scale the boost so it's significant but doesn't dominate the UCB term 
    # in the early stages.
    # In late stages (high progress), we want to trust the mean more.
    boost_weight = 0.1 + 0.4 * progress
    exploit_boost = boost_weight * np.log1p(improvement_potential / 10.0)
    
    scores += exploit_boost
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
