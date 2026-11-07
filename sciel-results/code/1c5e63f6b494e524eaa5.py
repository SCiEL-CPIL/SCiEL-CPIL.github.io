# EVOLVE-BLOCK-START
import numpy as np
import time
import warnings
from scipy.stats import norm
warnings.filterwarnings('ignore')

_start_time = None

def _get_elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition using Expected Improvement (EI).
    
    Strategy:
    1. Compute the Expected Improvement (EI) for each candidate.
       EI(x) = E[max(0, f_best - f(x))]
       For f(x) ~ N(mu, sigma^2), this is (f_best - mu)*Phi(z) + sigma*phi(z)
       where z = (f_best - mu) / sigma.
    2. Divide by the cost to get a cost-efficiency score.
       score = EI / cost
       
    This is a standard Bayesian expected utility acquisition function.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety checks for inputs
    sigma = np.maximum(sigma, 1e-8)
    sigma = np.where(np.isfinite(sigma), sigma, 1.0)
    
    cost = np.maximum(cost, 1e-8)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    
    mu = np.where(np.isfinite(mu), mu, 0.0)
    
    # Calculate z-score
    # Avoid division by zero, though sigma is clamped
    z = (f_best - mu) / sigma
    
    # Calculate EI
    # Phi(z) is the CDF, phi(z) is the PDF
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Note: if mu > f_best, z is negative. 
    # As z -> -inf, cdf -> 0, pdf -> 0. EI -> 0.
    # As z -> +inf, cdf -> 1, pdf -> 0. EI -> f_best - mu.
    ei = (f_best - mu) * cdf_z + sigma * pdf_z
    
    # Ensure EI is non-negative (numerical errors can cause tiny negatives)
    ei = np.maximum(ei, 0.0)
    
    # Cost-adjusted score
    score = ei / cost
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score

# EVOLVE-BLOCK-END
