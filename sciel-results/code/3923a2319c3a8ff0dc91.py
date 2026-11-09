import numpy as np
import time
from scipy.stats import norm

# Global start time for deadline guard
_START_TIME = time.time()

def acquisition(mu, sigma, f_best):
    """
    Computes acquisition scores using Expected Improvement (EI) for minimization.
    
    For a Gaussian posterior with mean mu and std sigma, and current best f_best:
      z = (f_best - mu) / sigma
      EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    
    This is the expected value of max(0, f_best - f(x)) under the GP posterior.
    Higher EI means the point is more likely to improve on the current best.
    """
    # Deadline guard: 330s limit. Return immediately if close to limit.
    # 10s safety margin.
    if time.time() - _START_TIME > 320:
        return np.zeros_like(mu)
    
    # Clip sigma to avoid division by zero and numerical issues
    sigma = np.maximum(sigma, 1e-12)
    
    # Compute z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    
    # Clip z to prevent extreme values which can cause numerical instability
    # in norm.pdf/cdf. z > 10 is effectively 1 for CDF and ~0 for PDF.
    # z < -10 is effectively 0 for CDF and very small for PDF.
    z = np.clip(z, -10, 10)
    
    # Compute EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Use scipy.stats for numerical stability
    Phi_z = norm.cdf(z)
    phi_z = norm.pdf(z)
    
    ei = (f_best - mu) * Phi_z + sigma * phi_z
    
    # EI should be non-negative, but due to numerical errors it might be slightly negative.
    # Clip to 0.
    ei = np.maximum(ei, 0.0)
    
    # Add a tiny exploitation term to break ties when sigma is very small.
    # This ensures that among points with near-zero sigma, the one with
    # the lowest mu (most promising) gets a slightly higher score.
    # The coefficient is small enough not to disturb the EI ranking in 
    # normal operation but large enough to resolve ties.
    eps = 1e-6
    scores = ei + eps * (-mu)
    
    # Ensure scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
