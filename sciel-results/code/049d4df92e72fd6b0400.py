import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 290  # 290s safety margin before 300s limit

def acquisition(mu, sigma, f_best, cost):
    # Check deadline - if we're close to the limit, return a simple deterministic score
    if time.time() > _DEADLINE:
        # Return scores that favor low cost (cheaper evaluations)
        return 1.0 / (1.0 + cost)
    
    # Clip sigma to avoid division by zero and numerical instability
    sigma = np.clip(sigma, 1e-9, None)
    
    # Calculate z-score
    z = (f_best - mu) / sigma
    
    # Calculate Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0.0)
    
    # Multiplicative gate based on signal-to-noise ratio
    # Score = EI * exp(-k * cost) where k is calibrated to the EI scale
    # Use k=2.0 to strongly suppress high-cost points unless EI is large
    k = 2.0
    gate = np.exp(-k * (cost - 1.0))  # Normalize so min cost has gate=1
    
    scores = ei * gate
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
