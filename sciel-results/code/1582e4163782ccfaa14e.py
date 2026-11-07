"""Cost-aware acquisition function for Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Expected Improvement (EI) acquisition function with adaptive
    exploration and cost normalization.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Mean predictions from the GP in objective units.
    sigma : array-like, shape (2000,)
        Standard deviation predictions from the GP in objective units.
    f_best : float
        Best observed objective value so far (minimum, since we minimize).
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate point, in [1, 2].
    
    Returns:
    --------
    score : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is strictly positive for numerical stability
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Ensure cost is strictly positive
    cost_safe = np.maximum(cost, 1.0)
    
    # Compute Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z), where z = (f_best - mu) / sigma
    # Phi is the standard normal CDF, phi is the standard normal PDF
    
    z = (f_best - mu) / sigma_safe
    
    # For numerical stability, handle large |z| values
    # When z is very large positive, Phi(z) ~ 1, phi(z) ~ 0
    # When z is very large negative, Phi(z) ~ 0, phi(z) ~ 0
    
    # Compute Phi(z) and phi(z) using scipy for accuracy
    from scipy.stats import norm
    
    Phi_z = norm.cdf(z)
    phi_z = norm.pdf(z)
    
    # Expected Improvement
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # Ensure EI is non-negative (numerical errors can make it slightly negative)
    ei = np.maximum(ei, 0.0)
    
    # Adaptive exploration: add a small UCB component to encourage exploration
    # This helps escape local minima in multimodal functions
    # Use a smaller kappa to balance exploitation and exploration
    kappa = 0.5
    ucb_component = kappa * sigma_safe
    
    # Combine EI and UCB: primarily exploit (EI) with some exploration (UCB)
    combined_score = ei + ucb_component
    
    # Cost-aware normalization: divide by cost to prefer cheaper evaluations
    # Use sqrt(cost) instead of cost to soften the cost penalty
    # This allows more exploration in expensive regions when the improvement is high
    score = combined_score / np.sqrt(cost_safe)
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Return the acquisition scores
    return score
# EVOLVE-BLOCK-END
