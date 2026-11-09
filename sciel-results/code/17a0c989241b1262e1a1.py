"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global start time for deadline tracking
_start_time = time.time()
_DEADLINE = 320.0  # 10s margin before 330s limit

def acquisition(mu, sigma, f_best):
    """
    Hybrid acquisition function: 0.7 * PI + 0.3 * Normalized_EI
    
    This combines the exploitation bias of Probability of Improvement (PI) 
    with the continuous gradient information of Expected Improvement (EI).
    EI is normalized to [0, 1] to ensure comparable scales with PI.
    """
    # Check deadline
    if time.time() > _DEADLINE:
        # Fallback: Greedy (minimize mu) is safe and fast
        # Score should be higher for better candidates, so -mu
        return -mu

    # Ensure sigma is positive and not zero to avoid division errors
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute z-score for improvement
    z = (f_best - mu) / sigma_safe
    
    # Compute Probability of Improvement (PI)
    # PI = P(f(x) < f_best) = Phi(z)
    pi = norm.cdf(z)
    
    # Compute Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Normalize EI to [0, 1] range
    # This ensures that the 0.3 weight is meaningful and not dominated by the absolute scale of EI
    ei_min = np.min(ei)
    ei_max = np.max(ei)
    
    if ei_max - ei_min > 1e-12:
        ei_norm = (ei - ei_min) / (ei_max - ei_min)
    else:
        # If EI is constant, all points are equivalent in EI terms
        ei_norm = np.zeros_like(ei)
        
    # Combine scores
    # Weight 0.7 on PI (exploitation/probability) and 0.3 on EI (magnitude of improvement)
    weight_pi = 0.7
    weight_ei = 0.3
    
    scores = weight_pi * pi + weight_ei * ei_norm
    
    # Ensure all scores are finite
    if not np.all(np.isfinite(scores)):
        scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
        
    return scores
# EVOLVE-BLOCK-END
