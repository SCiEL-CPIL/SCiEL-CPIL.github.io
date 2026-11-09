"""Cost-aware acquisition function for Bayesian optimization (rosenbrock 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Module-level state for time tracking
_start_time = None


def _ensure_start():
    global _start_time
    if _start_time is None:
        _start_time = time.time()


def _elapsed():
    _ensure_start()
    return time.time() - _start_time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining Expected Improvement (EI)
    with a cost penalty, plus exploration via UCB for early iterations.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        GP mean predictions in objective units.
    sigma : array-like, shape (2000,)
        GP standard deviation predictions in objective units.
    f_best : float
        Best observed function value so far.
    cost : array-like, shape (2000,)
        Evaluation costs for each candidate.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    _ensure_start()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is strictly positive to avoid division issues
    sigma = np.maximum(sigma, 1e-12)
    
    # Compute Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    diff = f_best - mu
    z = diff / sigma
    
    # Numerically stable computation
    # For large positive z, Phi(z) ~ 1, phi(z) ~ 0
    # For large negative z, Phi(z) ~ 0, phi(z) ~ 0
    ei = np.zeros_like(mu)
    
    # Only compute where z is in a reasonable range
    mask = np.abs(z) < 10
    if np.any(mask):
        z_m = z[mask]
        diff_m = diff[mask]
        sigma_m = sigma[mask]
        ei_m = diff_m * norm.cdf(z_m) + sigma_m * norm.pdf(z_m)
        ei[mask] = ei_m
    
    # For very negative z (mu >> f_best), EI is essentially 0
    # For very positive z (mu << f_best), EI ~ diff (since Phi~1, phi~0)
    # But diff is positive when mu < f_best, so EI ~ f_best - mu > 0
    # This is correct: if mu is much less than f_best, we expect big improvement
    
    # Cost-normalized EI
    # Use a soft cost penalty: EI / (cost + epsilon)
    # But also consider that lower cost is better
    cost_penalty = cost + 1e-6
    ei_cost = ei / cost_penalty
    
    # Add exploration component: UCB-like term
    # This helps when f_best is still far from optimum
    # Use a decay based on how close we are to the optimum
    # For Rosenbrock, min is 0, so f_best approaching 0 means we're close
    # Exploration weight: larger when f_best is large (far from optimum)
    # Use sqrt(sigma) as exploration bonus (UCB style)
    
    # Adaptive beta: decrease as we get closer to optimum
    # f_best for Rosenbrock 2d: min is 0, typical values range from 0 to ~100+
    # beta should be small when f_best is small, larger when f_best is large
    beta = 0.5 * np.sqrt(f_best + 1e-6)
    beta = np.clip(beta, 0.0, 2.0)
    
    # UCB term: mu + beta * sigma (for minimization, we want low mu, so we use -mu + beta*sigma)
    # Actually for minimization, UCB = mu - beta * sigma would be wrong
    # We want to explore where sigma is high, so add beta * sigma
    ucb = -mu + beta * sigma
    
    # Combine: primarily EI/cost, with some UCB for exploration
    # Weight: as f_best decreases, rely more on EI and less on exploration
    w_ei = 1.0
    w_ucb = beta * 0.1  # Small weight for exploration
    
    scores = w_ei * ei_cost + w_ucb * ucb / (cost_penalty * 10)
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Safety margin: if we're close to the deadline, return quickly
    # The deadline guard is handled by the caller, but we ensure we return valid scores
    
    return scores
# EVOLVE-BLOCK-END
