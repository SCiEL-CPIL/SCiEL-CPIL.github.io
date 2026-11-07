import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = _start_time + 300.0

# Track number of acquisition calls to adapt kappa
_call_count = 0

def acquisition(mu, sigma, f_best):
    """
    Compute acquisition scores for 2000 candidate points.
    Uses adaptive GP-UCB with a time-varying kappa for exploration/exploitation balance.
    """
    global _call_count
    _call_count += 1
    
    # Deadline guard
    if time.time() > _DEADLINE - 10.0:
        sigma = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma
        cdf_z = norm.cdf(z)
        pdf_z = norm.pdf(z)
        ei = (f_best - mu) * cdf_z + sigma * pdf_z
        return np.where(np.isfinite(ei), ei, 0.0)
    
    sigma = np.maximum(sigma, 1e-9)
    
    # Adaptive kappa: start high for exploration, decay to low for exploitation
    # Total evaluations ~30, so _call_count goes from 1 to 25 (30 - 5 initial)
    # Early: kappa ~ 2.0, Late: kappa ~ 0.3
    total_expected_calls = 25
    progress = min(_call_count / total_expected_calls, 1.0)
    
    # Use a smooth decay: kappa = kappa_min + (kappa_max - kappa_min) * (1 - progress)^2
    kappa_max = 2.0
    kappa_min = 0.3
    kappa = kappa_min + (kappa_max - kappa_min) * (1.0 - progress) ** 2
    
    # GP-UCB for minimization: score = -mu + kappa * sigma
    # Higher score means more promising (lower predicted value, higher uncertainty)
    score = -mu + kappa * sigma
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
