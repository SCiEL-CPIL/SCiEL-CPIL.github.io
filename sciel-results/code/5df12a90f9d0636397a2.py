# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    # Deadline guard: ensure we return within time budget
    t_start = time.time()
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Standard EI: (f_best - mu) * Phi(z) + sigma * phi(z)
    z = (f_best - mu) / sigma
    
    # Use scipy.stats.norm for numerical stability
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_z + sigma * pdf_z
    
    # Add a small exploration bonus to encourage diversity
    # The bonus is proportional to sigma but scaled down by distance from f_best
    # to avoid pure random exploration
    diff = np.abs(f_best - mu)
    bonus = 0.05 * sigma / (diff + sigma + 1e-6)
    
    score = ei + bonus
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, -1e10)
    
    # Check if we're close to the deadline; if so, just return
    elapsed = time.time() - t_start
    if elapsed > 280:  # 300s limit with safety margin
        return score
    
    return score
# EVOLVE-BLOCK-END
