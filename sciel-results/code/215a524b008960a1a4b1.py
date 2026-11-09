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
    
    Computes the expected reduction in the objective function per unit cost.
    ER = E[max(0, f_best - f(x))] / cost
    For a Gaussian posterior f(x) ~ N(mu, sigma^2), this has a closed form.
    
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
    # For f(x) ~ N(mu, sigma^2):
    # ER = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    # where Phi is the standard normal CDF and phi is the standard normal PDF.
    
    diff = f_best - mu
    z = diff / sigma_safe
    
    # Handle numerical stability for extreme z values
    # When z is very large positive, Phi(z) ~ 1 and phi(z) ~ 0, so ER ~ diff
    # When z is very negative, Phi(z) ~ 0 and phi(z) ~ 0, so ER ~ 0
    
    # Use stable computation
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    # Clip z to avoid numerical issues in extreme cases
    z_clipped = np.clip(z, -10, 10)
    phi_z = norm.pdf(z_clipped)
    Phi_z = norm.cdf(z_clipped)
    
    er = diff * Phi_z + sigma_safe * phi_z
    
    # Ensure ER is non-negative (numerical safety)
    er = np.maximum(er, 0.0)
    
    # Score: expected regret per unit cost
    scores = er / cost_safe
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
