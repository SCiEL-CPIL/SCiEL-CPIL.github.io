"""Cost-aware acquisition function for Bayesian optimization on rastrigin 2d."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_DEADLINE = 280.0
_start_time = time.time()


def _check_deadline():
    return (time.time() - _start_time) > _DEADLINE


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Expected Regret (ER) acquisition for minimizing rastrigin 2d.
    
    Computes the expected regret E[max(0, f_best - f(x))] / cost(x),
    which directly optimizes the expected reduction in the objective
    per unit evaluation cost. This is more principled than EI for
    cost-aware BO since it weights high-uncertainty regions by their
    actual expected benefit rather than a fixed improvement threshold.
    
    Parameters:
    -----------
    mu : array_like, shape (2000,)
        GP mean predictions.
    sigma : array_like, shape (2000,)
        GP standard deviation predictions.
    f_best : float
        Best observed value so far.
    cost : array_like, shape (2000,)
        Cost of evaluating each candidate (1 + mean(x_scaled)).
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means evaluate next.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    cost_safe = np.maximum(cost, 1e-6)
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Regret: E[max(0, f_best - f(x))]
    # where f(x) ~ N(mu, sigma^2)
    # ER = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    
    # Handle the case where sigma is very small (near-deterministic prediction)
    # In that case, ER ≈ max(0, f_best - mu)
    
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # For numerical stability, clip z to avoid overflow in exp
    z_clipped = np.clip(z, -10.0, 10.0)
    
    # Phi(z) = CDF of standard normal
    Phi_z = norm.cdf(z_clipped)
    
    # phi(z) = PDF of standard normal
    phi_z = norm.pdf(z_clipped)
    
    # Expected Regret
    er = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # Ensure non-negative (numerical issues can cause small negatives)
    er = np.maximum(er, 0.0)
    
    # Cost-aware score: expected regret per unit cost
    scores = er / cost_safe
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
