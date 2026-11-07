"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining Expected Improvement (EI)
    with a Knowledge Gradient (KG) term to balance exploitation and exploration.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        GP mean predictions at candidate points.
    sigma : array-like, shape (2000,)
        GP standard deviation predictions at candidate points.
    f_best : float
        Best observed value so far.
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher is better).
    """
    # Ensure inputs are numpy arrays for numerical stability
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Expected Improvement (EI)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Knowledge Gradient (KG) approximation:
    # The expected reduction in the optimal value is approximated by:
    # E[max(f_best, f(x))] - f_best = E[max(0, f(x) - f_best)]
    # For a Gaussian f(x) ~ N(mu, sigma^2):
    # E[max(0, f(x) - f_best)] = (mu - f_best) * Phi((mu - f_best)/sigma) + sigma * phi((mu - f_best)/sigma)
    # where Phi and phi are the standard normal CDF and PDF.
    
    z_kg = (mu - f_best) / sigma_safe
    kg = (mu - f_best) * norm.cdf(z_kg) + sigma_safe * norm.pdf(z_kg)
    
    # Combine EI and KG with a weight that emphasizes exploration early on.
    # The cost-aware division encourages efficient use of the budget.
    # We use a simple weighted sum: EI + lambda * KG, where lambda is a small constant.
    # The cost-aware version divides by cost to get "improvement per unit cost".
    
    lambda_kg = 0.5  # Weight for the KG term
    
    scores = (ei + lambda_kg * kg) / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
