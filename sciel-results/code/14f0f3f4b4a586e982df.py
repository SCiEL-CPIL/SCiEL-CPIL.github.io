import numpy as np
import time
from scipy.stats import norm

# Global deadline guard
_start_time = time.time()
_DEADLINE = _start_time + 320  # 320s out of 330s budget

def acquisition(mu, sigma, f_best, cost):
    # Check deadline
    if time.time() > _DEADLINE:
        return -mu / (cost + 1e-9)
    
    # Ensure inputs are clean
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.maximum(sigma, 1e-9)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Structural Change: Cost-Aware Expected Improvement (CA-EI)
    # EI is generally better than UCB for optimization as it directly maximizes expected gain.
    # Dividing by cost prioritizes high-gain, low-cost evaluations.
    
    # Calculate Z-score
    z = (f_best - mu) / sigma
    
    # Clip Z to avoid numerical issues in norm.cdf/pdf
    z = np.clip(z, -10, 10)
    
    # Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_z + sigma * pdf_z
    
    # Ensure EI is non-negative (numerical safety)
    ei = np.maximum(ei, 0)
    
    # Cost-aware score
    score = ei / cost
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, -1e10)
    
    return score
