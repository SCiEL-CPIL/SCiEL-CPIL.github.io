"""Rewritten program with hybrid UCB-EI acquisition for Weierstrass 1D BO."""
import numpy as np
from scipy.stats import norm
import time

# Deadline guard: evaluator kills at 330s, use 320s as safety margin
_START_TIME = time.time()
_DEADLINE = _START_TIME + 320.0


def _time_remaining():
    return _DEADLINE - time.time()


def acquisition(mu, sigma, f_best):
    """
    Hybrid acquisition: Pareto-filtered UCB with EI fallback.
    
    Parameters:
    -----------
    mu : np.ndarray, shape (2000,)
        GP mean predictions in objective units.
    sigma : np.ndarray, shape (2000,)
        GP standard deviation predictions in objective units.
    f_best : float
        Best observed value so far (lowest).
    
    Returns:
    --------
    scores : np.ndarray, shape (2000,)
        Acquisition scores; higher means more likely to be selected.
    """
    # Time guard: if we're close to deadline, return simple EI quickly
    if _time_remaining() < 5.0:
        z = (f_best - mu) / (sigma + 1e-9)
        return (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    n = len(mu)
    
    # Step 1: Compute EI as baseline
    z_ei = (f_best - mu) / (sigma + 1e-9)
    ei_scores = (f_best - mu) * norm.cdf(z_ei) + sigma * norm.pdf(z_ei)
    
    # Step 2: Pareto frontier filtering
    # Mask out candidates whose predicted mean is worse than f_best by more than tolerance
    # Use a tolerance that scales with the spread of f_best
    # Weierstrass 1D minimum is 0, so f_best should be close to 0
    # Tolerance: allow candidates where mu < f_best + tolerance
    # Choose tolerance based on the scale of the problem
    # The function values range roughly from 0 to ~5 on [-0.5, 0.5]
    # Use a tolerance of 0.5 * (f_best - min(mu)) if f_best > min(mu), else 0.1
    mu_min = np.min(mu)
    if f_best > mu_min:
        tol = 0.5 * (f_best - mu_min)
    else:
        tol = 0.1
    
    # Filter: keep candidates where mu <= f_best + tol
    mask = mu <= (f_best + tol)
    
    # If too many candidates are filtered out, use a larger tolerance or skip filtering
    if np.sum(mask) < n * 0.1:  # Less than 10% remain
        # Fallback: use a more lenient tolerance
        tol = 1.0 * (f_best - mu_min) if f_best > mu_min else 0.5
        mask = mu <= (f_best + tol)
    
    if np.sum(mask) < n * 0.05:  # Still less than 5% remain
        # Give up on filtering, use EI
        return ei_scores
    
    # Step 3: Compute UCB on filtered subset
    # UCB = mu + beta * sigma
    # Choose beta adaptively: larger beta encourages more exploration
    # For Weierstrass 1D with 25 steps, use beta that balances exploration/exploitation
    # Start with beta = 2.0, which is moderate exploration
    beta = 2.0
    
    ucb_scores = np.full(n, -np.inf)
    ucb_scores[mask] = mu[mask] + beta * sigma[mask]
    
    # Step 4: Combine UCB and EI
    # Use the maximum of UCB and EI to get both exploration and exploitation
    # But we want to prioritize UCB in the filtered region
    # Normalize scores to comparable scale
    # Actually, let's use a weighted combination
    # Weight: 0.7 * UCB + 0.3 * EI in the filtered region, EI elsewhere
    
    combined = np.zeros(n)
    combined[mask] = 0.7 * ucb_scores[mask] + 0.3 * ei_scores[mask]
    combined[~mask] = ei_scores[~mask]
    
    # Ensure all scores are finite
    combined = np.where(np.isfinite(combined), combined, 0.0)
    
    # Time guard: if we've used too much time, return
    if _time_remaining() < 2.0:
        return combined
    
    return combined
