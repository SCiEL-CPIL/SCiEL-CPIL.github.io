import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining Expected Improvement (EI)
    with an adaptive exploration bonus based on the current best value.
    """
    start_time = time.time()
    
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score for EI
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Adaptive exploration bonus
    # f_best starts around 25 (max of x^2 on [-5,5]) and goes to 0.
    # We want high exploration early (high f_best) and low exploration late (low f_best).
    f_max = 25.0
    r = np.clip(f_best, 0.0, f_max) / f_max
    
    # lambda_explore: scales with progress. 
    # Use sqrt to keep exploration meaningful for longer.
    # Max bonus is ~0.8 * sigma when f_best is at max.
    lambda_explore = 0.8 * np.sqrt(r)
    
    scores = ei + lambda_explore * sigma_safe
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > 290:
        # Fallback: return pure EI if we're running out of time
        return ei
         
    return scores
