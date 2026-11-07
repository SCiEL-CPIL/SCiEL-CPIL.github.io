import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Expected Improvement (EI) for minimization.
    """
    elapsed = time.time() - _start_time
    if elapsed > 325.0:
        # Fallback to pure exploitation if close to deadline
        return -np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)

    # Ensure sigma is strictly positive for division
    # If sigma is 0, the improvement is deterministic: max(f_best - mu, 0)
    safe_sigma = np.where(sigma > 1e-10, sigma, 1.0)
    
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / safe_sigma
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Note: For minimization, we want to improve f_best, so improvement is f_best - f(x).
    # If f(x) < f_best, improvement is positive.
    # The formula above is for maximization of improvement (f_best - f(x)).
    
    # Phi(z) = norm.cdf(z)
    # phi(z) = norm.pdf(z)
    
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = (f_best - mu) * Phi_z + safe_sigma * phi_z
    
    # Where sigma was effectively 0, EI should be max(f_best - mu, 0)
    # The formula above with sigma=1 and z=(f_best-mu) gives:
    # (f_best-mu)*Phi(f_best-mu) + 1*phi(f_best-mu)
    # If f_best > mu (z>0), Phi(z) is close to 1, phi(z) is small. Result ~ f_best-mu.
    # If f_best < mu (z<0), Phi(z) is close to 0, phi(z) is small. Result ~ 0.
    # So the formula naturally handles the limit well if we just use safe_sigma.
    # However, if sigma is truly 0, we should not add the sigma*phi term which is 0 anyway.
    # Actually, if sigma is 0, z is huge.
    # Let's just mask the result for very small sigma to be safe.
    mask = sigma > 1e-10
    ei_final = np.zeros_like(mu)
    ei_final[mask] = (f_best - mu[mask]) * norm.cdf((f_best - mu[mask]) / sigma[mask]) + sigma[mask] * norm.pdf((f_best - mu[mask]) / sigma[mask])
    ei_final[~mask] = np.maximum(f_best - mu[~mask], 0.0)
    
    # Ensure finite scores
    ei_final = np.nan_to_num(ei_final, nan=0.0, posinf=1e6, neginf=0.0)
    
    return ei_final
