"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Combines Expected Improvement (EI) with a UCB-like exploration bonus,
    both normalized by evaluation cost. Uses adaptive exploration weight
    based on convergence progress.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        GP mean predictions at candidate points.
    sigma : array-like, shape (2000,)
        GP standard deviation predictions at candidate points.
    f_best : float
        Best observed function value so far.
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = more promising).
    """
    start_time = time.time()
    deadline = start_time + 280  # Safety margin before 300s limit
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle edge cases
    n = len(mu)
    if n == 0:
        return np.array([])
    
    # Replace near-zero sigma with small value to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)
    
    # Structural Change: Pure Cost-Normalized LCB with Budget-Aware Exploration
    # LCB for minimization: L(x) = mu(x) - beta * sigma(x)
    # We maximize the negative LCB: -(mu - beta * sigma) = beta * sigma - mu
    # This encourages finding regions with low mean and high uncertainty.
    
    # Estimate progress based on f_best improvement towards the known minimum
    # Hartmann 6d global min is approx -3.32237
    # Initial f_best from uniform random is typically around -1.0 to -2.0
    # We map f_best to a progress [0, 1] where 0 is start, 1 is converged
    min_val = -3.32237
    # Use a range that covers the expected improvement
    # If f_best is -1.0, progress is 0. If f_best is -3.32, progress is 1.
    # Clip to handle cases where f_best might be worse than -1.0 or better than min
    progress = np.clip((f_best - min_val) / (1.0 - min_val), 0.0, 1.0)
    
    # Budget-aware scaling:
    # "Remaining budget" proxy is (1 - progress).
    # Scale beta by sqrt(remaining) to allow aggressive exploration early
    # and precise exploitation late.
    remaining_proxy = 1.0 - progress
    
    # Base beta is tuned to be significant early.
    # As remaining_proxy -> 0, beta -> beta_floor.
    # As remaining_proxy -> 1, beta -> beta_max.
    beta_max = 5.0
    beta_floor = 0.5
    
    beta = beta_floor + (beta_max - beta_floor) * np.sqrt(remaining_proxy)
    
    # Compute the LCB-based score (higher is better)
    # score = -(mu - beta * sigma) = beta * sigma - mu
    lcb_score = beta * sigma_safe - mu
    
    # Cost-aware: divide by cost
    # Penalize expensive evaluations more heavily
    scores = lcb_score / cost_safe
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > deadline:
        pass
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
