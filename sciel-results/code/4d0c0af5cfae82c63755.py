import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization (minimization).
    Uses a combination of Expected Improvement (EI) and an exploration term.
    
    For minimization:
    - EI is defined as E[max(f_best - f(x), 0)]
    - We want to maximize EI to find points likely to improve on f_best.
    
    To also encourage exploration in early stages or to escape local minima
    in the multimodal Weierstrass function, we add a small exploration bonus
    proportional to the uncertainty (sigma).
    
    The final score is EI + epsilon * sigma, where epsilon is a small constant.
    This ensures that points with high uncertainty get a slight boost, but the
    primary driver is the expected improvement.
    
    Parameters:
    mu (np.ndarray): Posterior mean at candidate points, shape (N,).
    sigma (np.ndarray): Posterior standard deviation at candidate points, shape (N,).
    f_best (float): Best observed function value so far.
    
    Returns:
    np.ndarray: Scores for each candidate point, shape (N,). Higher score is better.
    """
    # Ensure sigma is non-negative and has a small lower bound to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standardize the improvement: z = (f_best - mu) / sigma
    # If mu < f_best, z > 0, indicating a likely improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (EI) for minimization:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is the CDF of the standard normal distribution
    # and phi is the PDF of the standard normal distribution
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    EI = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # Add a small exploration term to encourage visiting uncertain regions
    # The coefficient is small so that EI dominates, but it helps in early stages
    # or when the GP is very certain (sigma is small)
    epsilon = 0.1
    exploration_bonus = epsilon * sigma_safe
    
    scores = EI + exploration_bonus
    
    # Ensure finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
