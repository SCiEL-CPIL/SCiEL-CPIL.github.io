"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_deadline = _start_time + 285.0

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of ackley 2d.
    Uses UCB-style acquisition with adaptive exploration coefficient:
      score = -mu + kappa * sigma
    where kappa decays over the optimization run to shift from exploration
    to exploitation. This is structurally different from EI-based methods.
    """
    # Deadline guard: if we are running out of time, fall back to greedy
    if time.time() > _deadline:
        return -mu

    n = len(mu)
    
    # Ensure sigma is non-negative and finite
    sigma = np.maximum(sigma, 0.0)
    sigma = np.nan_to_num(sigma, nan=0.0, posinf=0.0, neginf=0.0)
    mu = np.nan_to_num(mu, nan=f_best, posinf=f_best, neginf=f_best)
    
    # Estimate how far into the optimization we are using f_best as a proxy.
    # Ackley 2D minimum is 0. If f_best is close to 0, we're in exploitation phase.
    # If f_best is still large, we're in exploration phase.
    # Use a simple heuristic: progress = 1 - f_best / (f_best + 1)
    # This gives 0 when f_best is very large, 1 when f_best is near 0.
    progress = 1.0 - f_best / (f_best + 1.0)
    progress = np.clip(progress, 0.0, 1.0)
    
    # Adaptive kappa: start with high exploration (kappa=3.0), decay to low (kappa=0.5)
    # Use exponential decay for smooth transition
    kappa = 3.0 - 2.5 * progress
    
    # UCB score: higher is better
    # -mu rewards low predicted values (exploitation)
    # kappa * sigma rewards high uncertainty (exploration)
    scores = -mu + kappa * sigma
    
    # Add a small exploitation bonus for points already predicted below f_best
    # This reinforces convergence once we find good regions
    below_best = np.maximum(0.0, f_best - mu)
    scores = scores + 0.1 * below_best
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
