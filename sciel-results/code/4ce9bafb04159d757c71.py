import numpy as np
import time

# Global start time for deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 310.0  # 310s limit to leave margin before 330s kill

def acquisition(mu, sigma, f_best):
    # Deadline guard
    if time.time() > _DEADLINE:
        return np.zeros_like(mu)

    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-12)

    # Expected Improvement for minimization
    # EI(x) = E[max(f_best - f(x), 0)]
    # For GP: EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    # where Phi is the standard normal CDF and phi is the standard normal PDF
    
    z = (f_best - mu) / sigma
    # Clip z to avoid numerical issues
    z = np.clip(z, -10, 10)
    
    # Compute Phi(z) and phi(z)
    from scipy.stats import norm
    Phi = norm.cdf(z)
    phi = norm.pdf(z)
    
    EI = (f_best - mu) * Phi + sigma * phi
    
    # Ensure finite scores
    EI = np.where(np.isfinite(EI), EI, 0.0)
    
    return EI
