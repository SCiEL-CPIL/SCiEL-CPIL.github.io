"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = _start_time + 290.0  # Safety margin before 300s limit


def acquisition(mu, sigma, f_best, cost):
    # Structural change: Probability of Improvement (PI) weighted by uncertainty confidence, normalized by cost.
    # PI = Phi(z) where z = (f_best - mu) / sigma
    # Confidence term: sigma / (sigma + epsilon) to weight exploration when uncertainty is high
    # This is a more robust, outlier-resistant measure than EI, especially in short-budget scenarios.
    
    # Handle zero sigma to avoid division by zero
    sigma_safe = np.where(sigma > 1e-9, sigma, 1e-9)
    
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement
    pi = norm.cdf(z)
    
    # Uncertainty confidence: higher when sigma is larger
    # Use a soft normalization to keep it bounded
    confidence = sigma_safe / (sigma_safe + 0.1)
    
    # Combine PI and confidence
    combined = pi * confidence
    
    # Cost-aware score: normalized by cost
    score = combined / (cost + 1e-9)
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
