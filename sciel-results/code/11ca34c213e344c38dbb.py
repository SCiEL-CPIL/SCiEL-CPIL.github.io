import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-Weighted Expected Improvement (CW-EI) acquisition function.
    
    This function computes the Expected Improvement (EI) for each candidate point
    given the GP posterior mean (mu) and standard deviation (sigma), and the
    current best observed value (f_best). It then divides the EI by the evaluation
    cost to prioritize points that offer high expected improvement per unit cost.
    
    EI(x) = (f_best - mu) * Phi((f_best - mu) / sigma) + sigma * phi((f_best - mu) / sigma)
    
    Score(x) = EI(x) / cost(x)
    """
    start_time = time.time()
    
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)
    
    # Safety checks
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=1e-8, posinf=1e6, neginf=1e-8)
    cost = np.nan_to_num(cost, nan=1.5, posinf=2.0, neginf=1.0)
    
    # Ensure sigma and cost are strictly positive to avoid division by zero or negative sigma
    sigma = np.maximum(sigma, 1e-8)
    cost = np.maximum(cost, 1e-8)
    
    # Calculate Z-score
    z = (f_best - mu) / sigma
    
    # Clip z to avoid numerical issues in CDF/PDF for extreme values
    z_clipped = np.clip(z, -10, 10)
    
    # Standard Normal CDF and PDF
    phi = norm.pdf(z_clipped)
    Phi = norm.cdf(z_clipped)
    
    # Expected Improvement
    # Note: If f_best - mu is large positive, EI is large. If negative, EI is small/zero.
    ei = (f_best - mu) * Phi + sigma * phi
    
    # Ensure EI is non-negative (numerical noise can make it slightly negative)
    ei = np.maximum(ei, 0.0)
    
    # Cost-Weighted EI
    score = ei / cost
    
    # Time guard
    elapsed = time.time() - start_time
    if elapsed > 280:
        # Fallback: simple exploitation based on mean, penalized by cost
        score = -mu / cost
    
    # Ensure finite
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
