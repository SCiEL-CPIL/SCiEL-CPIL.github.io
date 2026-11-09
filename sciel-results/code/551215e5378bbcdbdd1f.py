"""Cost-aware acquisition function for Bayesian optimization (rosenbrock 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Quantile-based probability-of-improvement per unit cost.
    
    Uses the probability that f(x) < f_best (i.e., P(mu + sigma*Z < f_best))
    normalized by evaluation cost. This decouples from noisy GP mean estimates
    and focuses on probability mass below f_best, which can navigate the
    Rosenbrock valley more efficiently by prioritizing regions with high
    relative improvement likelihood.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean predictions in objective units
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions in objective units
    f_best : float
        Lowest observed function value
    cost : ndarray, shape (2000,)
        Evaluation costs (1 + mean(x scaled to [0,1]))
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next)
    """
    # Ensure numerical stability
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standardized distance: how many sigma below f_best is the mean
    # z = (f_best - mu) / sigma
    # We want P(f(x) < f_best) and E[f_best - f(x) | f(x) < f_best]
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (EI) formula:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Note: This is the standard EI for minimization where improvement is f_best - f(x).
    # If mu > f_best, z is negative, Phi(z) is small, but (f_best - mu) is negative.
    # Wait, standard EI is defined for maximization usually.
    # For minimization, Improvement I(x) = max(0, f_best - f(x)).
    # E[I(x)] = (f_best - mu) * P(f(x) < f_best) + sigma * phi(z)
    # where z = (f_best - mu) / sigma.
    # Let's verify:
    # If mu << f_best (good point), z is large positive. Phi(z) ~ 1, phi(z) ~ 0.
    # EI ~ (f_best - mu) * 1 + 0 = f_best - mu. Correct.
    # If mu >> f_best (bad point), z is large negative. Phi(z) ~ 0, phi(z) ~ 0.
    # EI ~ 0. Correct.
    
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    # Calculate EI
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # EI should be non-negative, clip small negatives due to numerical errors
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware: divide by cost to get expected improvement per unit cost
    cost_safe = np.maximum(cost, 1e-9)
    scores = ei / cost_safe
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
