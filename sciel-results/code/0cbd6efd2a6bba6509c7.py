# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Expected Improvement (EI).
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Safety checks
    sigma = np.maximum(sigma, 1e-12)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-12)
    mu = np.where(np.isfinite(mu), mu, f_best)
    
    # Calculate Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z) where z = (f_best - mu) / sigma
    # Note: For minimization, we want to improve upon f_best, so the "improvement" is f_best - f(x).
    # E[f_best - f(x)] = (f_best - mu) * P(f(x) < f_best) + E[f_best - f(x) | f(x) >= f_best] * P(f(x) >= f_best)
    # Actually, standard EI for minimization:
    # Let Y = f_best - f(x). We want E[max(Y, 0)].
    # If f(x) ~ N(mu, sigma^2), then Y ~ N(f_best - mu, sigma^2).
    # E[max(Y,0)] = (f_best - mu) * Phi((f_best-mu)/sigma) + sigma * phi((f_best-mu)/sigma)
    
    z = (f_best - mu) / sigma
    # Clip z to avoid numerical issues
    z = np.clip(z, -10, 10)
    
    phi = norm.pdf(z)
    Phi = norm.cdf(z)
    
    ei = (f_best - mu) * Phi + sigma * phi
    
    # Ensure finite
    score = np.where(np.isfinite(ei), ei, 0.0)
    
    return score
# EVOLVE-BLOCK-END
