# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Deadline guard
    t_start = time.time()
    DEADLINE = t_start + 280  # 300s limit, 20s margin

    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Calculate the standardized improvement
    # We are minimizing, so improvement is f_best - mu
    diff = f_best - mu
    z = diff / sigma_safe
    
    # Use scipy.stats.norm for vectorized CDF and PDF calculation
    # This is more robust and faster than np.vectorize(math.erf)
    cdf_val = norm.cdf(z)
    pdf_val = norm.pdf(z)
    
    # Expected Improvement (EI)
    # EI = (f_best - mu) * P(mu < f_best) + E[f_best - mu | mu < f_best] * P(mu < f_best)
    # E[f_best - mu | mu < f_best] = sigma * phi(z) / Phi(z)
    # So EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = diff * cdf_val + sigma_safe * pdf_val
    
    # Ensure finite results
    ei[~np.isfinite(ei)] = 0.0
    
    # Check deadline
    if time.time() > DEADLINE:
         pass

    return ei
# EVOLVE-BLOCK-END
