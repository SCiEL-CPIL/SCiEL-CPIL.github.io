"""Cost-aware acquisition function for Bayesian optimization (styblinski tang 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Compute a cost-aware acquisition score for 2000 candidates.
    Uses a cost-aware Expected Improvement (EI) acquisition function.
    This is a structural change from the previous LCB approach, moving from
    confidence-bound exploration to improvement-based exploitation.
    
    Args:
        mu: Array of GP means (shape 2000,)
        sigma: Array of GP standard deviations (shape 2000,)
        f_best: Current best observed value (scalar)
        cost: Array of evaluation costs (shape 2000,)
        
    Returns:
        Array of acquisition scores (shape 2000,), higher is better.
    """
    start_time = time.time()
    deadline = 325.0  # Safety margin before 330s limit

    # Convert to numpy arrays and ensure float64
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Sanitize inputs
    # Replace non-finite mu with f_best (neutral)
    mu_safe = np.where(np.isfinite(mu), mu, f_best)
    # Replace non-finite or tiny sigma with a small positive value
    sigma_safe = np.where(np.isfinite(sigma) & (sigma > 1e-9), sigma, 1e-6)
    # Replace non-finite or tiny cost with 1.0 (minimum cost)
    cost_safe = np.where(np.isfinite(cost) & (cost > 1e-9), cost, 1.0)

    # Calculate Z-score for Expected Improvement
    # Z = (f_best - mu) / sigma
    # For minimization, we want f(x) < f_best.
    z = (f_best - mu_safe) / sigma_safe
    
    # Calculate PDF and CDF of standard normal distribution
    # norm.pdf(z) is phi(z)
    # norm.cdf(z) is Phi(z)
    phi_z = norm.pdf(z)
    phi_cdf = norm.cdf(z)
    
    # Expected Improvement for minimization:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu_safe) * phi_cdf + sigma_safe * phi_z
    
    # Cost-aware adjustment
    # We maximize the expected improvement per unit cost.
    score = ei / cost_safe
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Deadline check
    if time.time() > deadline:
         return score

    return score
# EVOLVE-BLOCK-END
