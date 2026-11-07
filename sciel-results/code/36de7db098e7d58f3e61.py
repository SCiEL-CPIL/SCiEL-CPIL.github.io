"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_module_start = time.time()
_DEADLINE = 280.0  # 20s safety margin before 300s limit


def acquisition(mu, sigma, f_best):
    # Deadline guard
    if time.time() - _module_start > _DEADLINE:
        return -mu.astype(float)
    
    # Thompson Sampling-inspired score:
    # We use the probability of improvement (POI) as a base, which is bounded [0,1]
    # and emphasizes candidates likely to improve over f_best.
    # POI = P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    
    # Handle case where sigma is very small (numerical stability)
    sigma_safe = np.where(sigma > 1e-10, sigma, 1e-10)
    
    # Standardized improvement
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement
    poi = norm.cdf(z)
    
    # Ensure all scores are finite and in [0, 1]
    scores = np.clip(poi, 0.0, 1.0)
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
