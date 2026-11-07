import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = _start_time + 300.0

def acquisition(mu, sigma, f_best):
    """
    Compute acquisition scores for 2000 candidate points.
    Uses a POI-based acquisition with uncertainty scaling for minimization.
    """
    # Deadline guard
    if time.time() > _DEADLINE - 10.0:
        sigma = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma
        cdf_z = norm.cdf(z)
        pdf_z = norm.pdf(z)
        ei = (f_best - mu) * cdf_z + sigma * pdf_z
        return np.where(np.isfinite(ei), ei, 0.0)

    sigma = np.maximum(sigma, 1e-9)
    
    # Probability of Improvement (POI) scaled by Uncertainty (Sigma)
    # This balances exploration (high sigma) and exploitation (high POI)
    # POI = P(f(x) < f_best) = Phi(z)
    # Score = POI * Sigma
    
    z = (f_best - mu) / sigma
    poi = norm.cdf(z)
    
    score = poi * sigma
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
