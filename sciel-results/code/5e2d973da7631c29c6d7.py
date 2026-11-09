import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Returns 2000 finite scores (higher = evaluate next).
    Uses Cost-Adjusted Expected Improvement (EI / Cost).
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    f_best = float(f_best)
    cost = np.asarray(cost, dtype=np.float64)

    # Ensure sigma is positive to avoid division by zero
    # A small epsilon prevents division by zero and numerical instability
    sigma = np.maximum(sigma, 1e-8)
    
    # Ensure cost is positive (it should be >= 1.0 based on problem description)
    cost = np.maximum(cost, 1e-8)

    # Calculate z-score for Expected Improvement
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    
    # Calculate Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Phi is the standard normal CDF, phi is the standard normal PDF
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_z + sigma * pdf_z
    
    # Cost-Adjusted EI
    # We want to maximize improvement per unit cost
    score = ei / cost
    
    # Handle any potential numerical issues (NaNs or Infs)
    # If EI is negative (which shouldn't happen for minimization with correct z), 
    # or if cost is 0, we might get issues. 
    # EI is always >= 0 for minimization if f_best is the best observed.
    # However, due to floating point errors, very small negative EI might occur if mu > f_best significantly?
    # No, if mu > f_best, z is negative. 
    # Let's check: if mu > f_best, z < 0. 
    # (f_best - mu) is negative. Phi(z) is small. sigma * phi(z) is positive.
    # Actually, EI is always non-negative.
    
    # Replace non-finite values with 0.0
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
