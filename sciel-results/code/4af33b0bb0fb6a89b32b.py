import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    start = time.time()
    limit = 2.0  # 2 seconds limit per call, plenty for N=2000
    
    # Ensure sigma is non-negative for safety
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Quantile Acquisition Function (QAF) / Probability of Improvement (POI)
    # Maximizes the probability that f(x) < f_best
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    
    z = (f_best - mu) / sigma_safe
    scores = norm.cdf(z)
    
    # If it takes too long (unlikely), fallback to simple greedy
    if time.time() - start > limit:
        scores = -mu
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
