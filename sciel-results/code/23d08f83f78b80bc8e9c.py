import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization (minimization).
    Uses Expected Improvement (EI) with a lower bound on sigma to handle
    potential underestimation of uncertainty by the GP in high-frequency regions
    of the Weierstrass function.
    
    Parameters:
    mu (np.ndarray): Posterior mean at candidate points, shape (N,).
    sigma (np.ndarray): Posterior standard deviation at candidate points, shape (N,).
    f_best (float): Best observed function value so far.
    
    Returns:
    np.ndarray: Scores for each candidate point, shape (N,). Higher score is better.
    """
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Weierstrass function is highly oscillatory. The GP (Matern 5/2) might
    # underestimate uncertainty in these regions. We enforce a minimum sigma
    # to encourage exploration even when the GP is "confident".
    # The scale of Weierstrass is roughly O(1). A floor of 0.05 is a heuristic
    # to prevent premature convergence to local minima.
    sigma_floor = 0.05
    sigma_eff = np.maximum(sigma_safe, sigma_floor)
    
    # Expected Improvement for minimization
    # EI(x) = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    
    # Calculate z
    z = (f_best - mu) / sigma_eff
    
    # Clip z to avoid numerical issues in norm.cdf/pdf for very large/small z
    z = np.clip(z, -10, 10)
    
    # Calculate EI
    ei = (f_best - mu) * norm.cdf(z) + sigma_eff * norm.pdf(z)
    
    # Ensure finite values
    ei = np.nan_to_num(ei, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # EI is non-negative. Higher EI is better.
    return ei
