import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Deadline guard: ensure we return within a safe margin
    start = time.time()
    limit = 1.0  # 1 second limit per call, plenty for N=2000
    
    # Ensure sigma is non-negative for safety
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Structural Change: Use Expected Improvement (EI) instead of UCB.
    # EI is a probabilistic acquisition function that balances exploration and exploitation
    # based on the probability of finding a point better than f_best.
    # For minimization, we want to maximize the expected reduction in the objective.
    
    # Calculate z-score: (f_best - mu) / sigma
    # If mu < f_best, z is positive, implying high probability of improvement.
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement formula:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Where Phi is the CDF of the standard normal distribution
    # and phi is the PDF of the standard normal distribution.
    
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    scores = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Check if computation took too long (unlikely for simple arithmetic)
    if time.time() - start > limit:
        # Fallback to a simple exploitation strategy if EI fails or is too slow
        scores = -mu
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
