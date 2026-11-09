import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        Mean of the GP posterior
    sigma : array of shape (2000,)
        Standard deviation of the GP posterior
    f_best : float
        Best (lowest) observed value
    cost : array of shape (2000,)
        Cost of evaluating each candidate
    
    Returns:
    --------
    score : array of shape (2000,)
        Acquisition scores (higher = more promising to evaluate next)
    """
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 300s limit
    
    eps = 1e-10
    sigma_safe = np.maximum(sigma, eps)
    cost_safe = np.maximum(cost, eps)
    
    # Use Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z) where z = (f_best - mu) / sigma
    # For minimization, we want to maximize the expected improvement over f_best
    
    # Compute z-score
    z = (f_best - mu) / sigma_safe
    
    # Standard normal CDF and PDF
    # Use scipy if available, otherwise use numpy approximations
    try:
        from scipy.stats import norm
        cdf_z = norm.cdf(z)
        pdf_z = norm.pdf(z)
    except ImportError:
        # Fallback: use error function approximation
        cdf_z = 0.5 * (1 + np.erf(z / np.sqrt(2)))
        pdf_z = np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi)
    
    # Expected Improvement for minimization
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware adjustment: divide by cost to favor cheaper evaluations
    # This makes the acquisition function prefer points that are both
    # likely to improve f_best AND have low evaluation cost
    cost_adjusted_ei = ei / cost_safe
    
    # Add a small exploration term to avoid getting stuck
    # UCB-like term with small weight
    beta = 0.1
    exploration = beta * sigma_safe / cost_safe
    
    score = cost_adjusted_ei + exploration
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Check if we're approaching the deadline
    if time.time() > deadline:
        # Return a simple fallback score
        return -mu / cost_safe
    
    return score
