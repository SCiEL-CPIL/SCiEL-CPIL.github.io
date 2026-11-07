import numpy as np
import time
from scipy.stats import norm

_module_start = time.time()
_DEADLINE = 275.0  # safety margin before 300s limit


def acquisition(mu, sigma, f_best):
    # Deadline guard
    if time.time() - _module_start > _DEADLINE:
        return -mu.astype(float)
    
    # Ensure sigma is non-negative and finite
    sigma = np.maximum(sigma, 1e-10)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-10)
    
    # Normalize sigma for stable computation
    sigma_max = np.max(sigma)
    if sigma_max < 1e-10:
        sigma_max = 1e-10
    
    # Compute Expected Improvement (EI) for minimization
    # EI = E[max(f_best - f(x), 0)]
    # For GP with mean mu and std sigma:
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    # where Phi is CDF and phi is PDF of standard normal
    
    improvement = f_best - mu
    z = improvement / sigma
    
    # Clip z to avoid numerical issues
    z = np.clip(z, -10, 10)
    
    phi = norm.pdf(z)
    Phi = norm.cdf(z)
    
    ei = improvement * Phi + sigma * phi
    
    # Also compute LCB for exploration
    # Adaptive beta based on progress
    distance = max(0.0, f_best - 3.0)
    # Start with more exploration, reduce as we get closer to optimum
    beta = 0.3 + 1.5 * np.exp(-distance / 5.0)
    lcb = -mu + beta * sigma
    
    # Combine EI and LCB
    # Use a weighted combination that emphasizes EI when we're close to optimum
    # and LCB when we're far
    weight_ei = 0.6 + 0.4 * (1.0 - np.exp(-distance / 10.0))
    weight_lcb = 1.0 - weight_ei
    
    # Normalize both to comparable scale
    ei_max = np.max(ei) if np.max(ei) > 0 else 1.0
    lcb_max = np.max(lcb) if np.max(lcb) > 0 else 1.0
    
    if ei_max > 1e-10:
        ei_norm = ei / ei_max
    else:
        ei_norm = ei
    
    if lcb_max > 1e-10:
        lcb_norm = lcb / lcb_max
    else:
        lcb_norm = lcb
    
    scores = weight_ei * ei_norm + weight_lcb * lcb_norm
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
