import numpy as np
import time
from scipy.stats import norm

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = 330.0  # seconds total budget, leave margin

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Hartmann 6D.
    Uses Expected Improvement (EI) for minimization.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure finite inputs
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    
    # Avoid division by zero
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    
    # Z-score for Expected Improvement
    # We are minimizing, so improvement is f_best - mu
    # If mu < f_best, potential improvement exists.
    improvement = f_best - mu
    z = improvement / sigma_safe
    
    # Standard Normal CDF and PDF
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    # EI formula: E[max(f_best - f(x), 0)]
    # = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = improvement * cdf + sigma_safe * pdf
    
    # Clip negative EI to 0 (mathematically EI >= 0, but numerical errors might occur)
    ei = np.maximum(ei, 0.0)
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(ei), ei, 0.0)
    
    return score
