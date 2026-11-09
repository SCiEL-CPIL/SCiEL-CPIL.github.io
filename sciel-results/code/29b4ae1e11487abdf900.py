"""Rewritten program using a Cost-aware Expected Improvement (EI) strategy."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Acquisition function for cost-aware Bayesian optimization.
    
    Uses Cost-aware Expected Improvement (EI).
    EI measures the expected reduction in the best observed value.
    Dividing by cost prioritizes evaluations that offer high improvement per unit cost.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean predictions of the GP
    sigma : ndarray, shape (2000,)
        Standard deviation predictions of the GP
    f_best : float
        Best observed value so far (minimum)
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next)
    """
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-6)
    
    # Calculate the potential improvement
    improvement = f_best - mu
    
    # Calculate the z-score for the improvement
    z = improvement / sigma_safe
    
    # Expected Improvement formula for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is the CDF and phi is the PDF of the standard normal distribution
    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost-aware score: normalize EI by the evaluation cost
    score = ei / cost_safe
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
