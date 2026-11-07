"""Rewritten program with adaptive cost-aware acquisition."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement (EI)
    with an adaptive exploration term and a softened cost penalty.
    
    Parameters:
    mu (np.ndarray): Mean of the GP posterior, shape (N,).
    sigma (np.ndarray): Std dev of the GP posterior, shape (N,).
    f_best (float): Best observed function value so far.
    cost (np.ndarray): Cost of evaluating each candidate, shape (N,).
    
    Returns:
    np.ndarray: Acquisition scores, shape (N,). Higher is better.
    """
    start_time = time.time()
    
    # Numerical stability epsilon
    eps = 1e-9
    
    # Ensure sigma is positive for division
    sigma_safe = np.maximum(sigma, eps)
    
    # Calculate Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Handle cases where sigma is very small (near certain)
    # In these cases, EI should be max(f_best - mu, 0)
    low_sigma_mask = sigma < 1e-5
    if np.any(low_sigma_mask):
        ei[low_sigma_mask] = np.maximum(f_best - mu[low_sigma_mask], 0)
        
    # Adaptive Exploration Term
    # Scale exploration beta with the average uncertainty of the model
    # Early on, high uncertainty -> high beta -> more exploration
    # Later, low uncertainty -> low beta -> more exploitation
    mean_sigma = np.mean(sigma)
    beta = 0.1 * mean_sigma
    
    # Structural Change: Softened Cost Penalty
    # Baseline used cost^-1. Insights suggest cost^-0.7 is more effective
    # as it allows considering slightly more expensive points with high EI.
    gamma = 0.7
    
    # Calculate final score
    cost_safe = np.maximum(cost, eps)
    scores = (ei + beta * sigma_safe) / (cost_safe ** gamma)
    
    # Deadline Guard: Ensure we return before 300s (using 290s margin)
    if time.time() - start_time > 290:
        # Return small random scores to ensure a valid selection
        return np.random.rand(len(mu)) * 1e-6
        
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
