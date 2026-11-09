import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function blending Expected Improvement and UCB.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean predictions
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best observed function value
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = more promising)
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure cost is positive and finite
    cost_safe = np.where(cost > 1e-10, cost, 1.0)
    
    # Expected Improvement
    sigma_safe = np.where(sigma > 1e-10, sigma, 1e-10)
    z = (f_best - mu) / sigma_safe
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    ei = (f_best - mu) * cdf_z + sigma * pdf_z
    ei = np.where(ei > 0, ei, 0.0)  # EI is non-negative
    
    # UCB for minimization: we want low mu, high sigma
    # score_ucb = -(mu - kappa * sigma) = -mu + kappa * sigma
    kappa = 0.6  # exploration-exploitation balance
    ucb = -mu + kappa * sigma
    
    # Normalize each component by cost
    ei_cost = ei / cost_safe
    ucb_cost = ucb / cost_safe
    
    # Blend: weight EI more heavily for exploitation, UCB for exploration
    # Use 0.65 EI + 0.35 UCB to favor exploitation but maintain exploration
    scores = 0.65 * ei_cost + 0.35 * ucb_cost
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
