import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
    mu (np.ndarray): Mean of the GP posterior, shape (2000,).
    sigma (np.ndarray): Standard deviation of the GP posterior, shape (2000,).
    f_best (float): The best (lowest) observed function value so far.
    cost (np.ndarray): Cost of evaluating each candidate, shape (2000,).
    
    Returns:
    np.ndarray: Acquisition scores, shape (2000,). Higher is better.
    """
    # Ensure sigma is positive to avoid division issues
    sigma = np.maximum(sigma, 1e-10)
    
    # Compute Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    
    # Use stable computation for EI
    # For very large z, cdf(z) -> 1, pdf(z) -> 0, EI -> f_best - mu
    # For very small z, cdf(z) -> 0, pdf(z) -> 0, EI -> 0
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0.0)
    
    # Add an exploration term to encourage visiting uncertain areas
    # This helps when EI is low but uncertainty is high
    beta = 1.0
    exploration_term = beta * sigma
    
    # Combine EI and exploration term
    # This creates a score that balances exploitation (EI) and exploration (uncertainty)
    combined_score = ei + exploration_term
    
    # Cost-aware adjustment: divide by cost
    # This prioritizes points that offer high expected benefit per unit cost
    result = combined_score / cost
    
    # Ensure all values are finite
    result = np.nan_to_num(result, nan=0.0, posinf=1e10, neginf=-1e10)
    
    return result
