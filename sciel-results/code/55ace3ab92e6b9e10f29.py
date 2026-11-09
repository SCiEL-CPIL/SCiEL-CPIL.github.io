"""Cost-aware acquisition for Ackley 2D Bayesian optimization."""
import time
import numpy as np
from scipy.stats import norm

# Deadline guard: each call should finish well within limits
_DEADLINE = 320.0  # safety margin before 330s kill


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function: normalized expected improvement
    with adaptive exploration for Ackley 2D minimization.
    
    Args:
        mu: array of GP mean predictions, shape (2000,)
        sigma: array of GP std predictions, shape (2000,)
        f_best: current best observed value (float)
        cost: array of evaluation costs, shape (2000,)
    
    Returns:
        array of acquisition scores, shape (2000,), higher = prefer
    """
    t_start = time.monotonic()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = len(mu)
    
    # Sanitize inputs
    sigma = np.maximum(sigma, 1e-10)
    sigma = np.nan_to_num(sigma, nan=1e-10, posinf=1e-2, neginf=1e-10)
    mu = np.nan_to_num(mu, nan=f_best, posinf=1e4, neginf=-1e4)
    cost = np.maximum(cost, 1e-9)
    cost = np.nan_to_num(cost, nan=1.0, posinf=1.0, neginf=1.0)
    
    if time.monotonic() - t_start > 25:
        # Fallback: simple cost-normalized UCB
        scores = (-mu + 2.0 * sigma) / cost
        scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
        return scores
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z), z = (f_best - mu)/sigma
    improvement = f_best - mu
    z = improvement / sigma
    ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure EI is non-negative (clip tiny negatives from numerical error)
    ei = np.maximum(ei, 0.0)
    ei = np.nan_to_num(ei, nan=0.0, posinf=1e4, neginf=0.0)
    
    # Structural Change: Regime-based exploration weight
    # High f_best -> Explore more (higher gamma)
    # Low f_best -> Exploit more (lower gamma)
    # This replaces the complex adaptive blending and normalization.
    if f_best > 2.0:
        gamma = 1.0
    elif f_best > 0.5:
        gamma = 0.5
    else:
        gamma = 0.2
        
    # Raw additive combination: (EI + gamma * sigma) / cost
    # This preserves the physical scale of improvement and uncertainty.
    scores = (ei + gamma * sigma) / cost
    
    # Ensure finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Check deadline
    if time.monotonic() - t_start > 28:
        return scores
    
    return scores
