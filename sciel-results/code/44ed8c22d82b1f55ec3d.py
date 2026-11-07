import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of branin 2d.
    
    Uses Expected Improvement (EI) for minimization, which is the standard
    and well-tuned acquisition for BO. EI naturally balances exploration and
    exploitation:
    - Where mu < f_best (promising), EI is high
    - Where sigma is high (uncertain), EI is boosted
    - Where mu >> f_best (bad), EI is near zero
    
    For Branin's multi-modal landscape, EI's inherent exploration via the
    sigma term helps discover all three global minima within 30 evaluations.
    """
    start_time = time.time()
    
    # Safety: handle zero/near-zero sigma
    eps = 1e-12
    sigma_safe = np.where(sigma > eps, sigma, eps)
    
    # Expected Improvement for minimization:
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    # where Phi is standard normal CDF, phi is standard normal PDF
    Z = (f_best - mu) / sigma_safe
    improvement = f_best - mu
    
    # EI formula
    ei = improvement * norm.cdf(Z) + sigma_safe * norm.pdf(Z)
    
    # Clamp to non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Ensure finiteness
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # Safety margin check
    elapsed = time.time() - start_time
    if elapsed > 280.0:
        return ei
    
    return ei
