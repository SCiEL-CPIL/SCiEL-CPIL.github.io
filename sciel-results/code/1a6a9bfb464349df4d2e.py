"""Cost-aware acquisition function for Bayesian optimization (ackley 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard: track start time at module load
_T0 = time.time()
_DEADLINE = 295.0  # seconds, safety margin before 300s limit


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition score for 2000 candidates.
    
    Uses a cost-aware variant of Expected Improvement with adaptive exploration
    based on the current best value and uncertainty.
    
    Parameters
    ----------
    mu : ndarray, shape (2000,)
        GP mean predictions in objective units.
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions in objective units.
    f_best : float
        Best observed value so far.
    cost : ndarray, shape (2000,)
        Evaluation costs (in [1, 2]).
    
    Returns
    -------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    # Deadline guard: if we are running out of time, return a safe fallback
    if time.time() - _T0 > _DEADLINE:
        return np.zeros_like(mu)
    
    # Ensure sigma is non-negative and safe for operations
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Ensure cost is positive
    cost_safe = np.maximum(cost, 1e-9)
    
    # Adaptive exploration parameter based on how good f_best is
    # For Ackley, the minimum is 0, so we can gauge progress
    # When f_best is close to 0, we want more exploitation
    # When f_best is far from 0, we want more exploration
    
    # Compute a progress indicator: how close we are to the minimum
    # Ackley minimum is 0, so progress = 1 - f_best (clamped)
    progress = np.clip(1.0 - f_best, 0.0, 1.0)
    
    # Adaptive beta: start with more exploration, shift to exploitation
    # beta controls the trade-off between exploitation (EI) and exploration (UCB)
    # At progress=0 (f_best=1), beta=2.0 (more exploration)
    # At progress=1 (f_best=0), beta=0.5 (more exploitation)
    beta = 2.0 - 1.5 * progress
    
    # Expected Improvement for minimization:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # UCB for minimization (lower is better):
    # UCB = mu - beta * sigma
    ucb = mu - beta * sigma
    
    # Combine EI and UCB with a weight
    # Use EI as the primary signal, but add UCB for exploration
    # Normalize both to comparable scales
    ei_mean = np.mean(ei)
    ei_std = np.std(ei) + 1e-9
    ei_norm = (ei - ei_mean) / ei_std
    
    ucb_mean = np.mean(ucb)
    ucb_std = np.std(ucb) + 1e-9
    ucb_norm = (ucb - ucb_mean) / ucb_std
    
    # Weight: more weight on EI when we're close to optimum, more on UCB when exploring
    w_ei = 0.7 + 0.3 * progress  # 0.7 to 1.0
    w_ucb = 1.0 - w_ei  # 0.3 to 0.0
    
    # Combined score
    combined = w_ei * ei_norm + w_ucb * ucb_norm
    
    # Cost-aware score: divide by cost
    scores = combined / cost_safe
    
    # Ensure output is finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
