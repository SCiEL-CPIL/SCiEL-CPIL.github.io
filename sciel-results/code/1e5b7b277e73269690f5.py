"""Cost-aware acquisition function for Bayesian optimization (powell 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Compute cost-aware acquisition scores for 2000 candidates.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean predictions
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions
    f_best : float
        Lowest observed function value
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate (1 + mean(x_scaled))
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next)
    """
    # Deadline guard: track elapsed time
    t_start = time.time()
    DEADLINE = 295.0  # Safety margin before 300s limit
    
    # Ensure inputs are numpy arrays with correct shape
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle edge cases where sigma is 0 or very small
    sigma_safe = np.where(sigma > 1e-9, sigma, 1e-9)
    
    # Check deadline
    if time.time() - t_start > DEADLINE:
        return np.zeros_like(mu)
    
    # Compute z-score for Expected Improvement (minimization)
    # Improvement is (f_best - mu) when mu < f_best
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Check deadline
    if time.time() - t_start > DEADLINE:
        return ei / cost
    
    # Add a UCB term to encourage exploration
    # UCB = beta * sigma, where beta is a balancing parameter
    # We use a moderate beta that adapts to the scale of the problem
    beta = 0.5
    ucb = beta * sigma_safe
    
    # Blend EI and UCB, both normalized by cost
    # The blend helps avoid premature exploitation of noisy low-mu regions
    ei_cost = ei / np.where(cost > 1e-9, cost, 1e-9)
    ucb_cost = ucb / np.where(cost > 1e-9, cost, 1e-9)
    
    # Combined score: weighted sum with EI as primary and UCB as secondary
    # Weight 0.8 for EI, 0.2 for UCB
    scores = 0.8 * ei_cost + 0.2 * ucb_cost
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check deadline
    if time.time() - t_start > DEADLINE:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
