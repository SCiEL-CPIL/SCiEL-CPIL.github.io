"""Cost-aware acquisition function for Bayesian optimization of Powell 4D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition score for minimization.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean predictions in objective units.
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions in objective units.
    f_best : float
        Best (lowest) observed objective value.
    cost : ndarray, shape (2000,)
        Marginal cost of evaluating each candidate: 1 + mean(x_scaled).
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means evaluate next.
    """
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # z-score for expected improvement (minimization)
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost-aware scoring:
    # - Exploitation term (EI) is divided by cost (strong cost penalty)
    # - Exploration term (sigma) is divided by sqrt(cost) (milder cost penalty)
    # This balances cost-efficiency with exploration in a tight budget.
    
    cost_safe = np.maximum(cost, 1e-10)
    
    # Split EI into exploitation and exploration components
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    exploit = (f_best - mu) * norm.cdf(z)
    explore = sigma_safe * norm.pdf(z)
    
    # Cost-adjusted score
    scores = exploit / cost_safe + explore / np.sqrt(cost_safe)
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
