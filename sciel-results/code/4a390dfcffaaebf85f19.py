import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-Weighted Expected Improvement (CW-EI) Acquisition Function.
    
    This function calculates the Expected Improvement (EI) of each candidate point
    relative to the current best value (f_best), and then normalizes this improvement
    by the cost of evaluation. This encourages the optimizer to find points that are
    likely to improve the best value efficiently (low cost).
    
    EI(x) = (f_best - mu(x)) * Phi(z) + sigma(x) * phi(z)
    where z = (f_best - mu(x)) / sigma(x)
    
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
    
    # Ensure sigma and cost are strictly positive
    sigma = np.maximum(sigma, 1e-8)
    cost = np.maximum(cost, 1.0)
    
    # Calculate z-score
    z = (f_best - mu) / sigma
    # Clip z to avoid numerical issues in norm.cdf and norm.pdf
    z_clipped = np.clip(z, -10, 10)
    
    # Calculate EI components
    # Phi(z) is norm.cdf(z)
    # phi(z) is norm.pdf(z)
    cdf_z = norm.cdf(z_clipped)
    pdf_z = norm.pdf(z_clipped)
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Note: (f_best - mu) can be negative, but EI is max(0, ...).
    # The formula (f_best - mu)Phi(z) + sigma*phi(z) is always non-negative.
    ei = (f_best - mu) * cdf_z + sigma * pdf_z
    
    # Ensure EI is non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Cost-Weighted EI
    score = ei / cost
    
    # Time guard
    elapsed = time.time() - start_time
    if elapsed > 280:
        # Fallback: simple exploitation
        # Use -mu / cost
        score = -mu / cost
    
    # Ensure finite
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
