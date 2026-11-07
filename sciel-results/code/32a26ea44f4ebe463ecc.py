"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    # Start time for deadline guard (though vectorized op is very fast)
    start_time = time.time()
    
    # Calculate z-score for Probability of Improvement
    # z = (f_best - mu) / sigma
    # We add a small epsilon to sigma to avoid division by zero
    eps = 1e-9
    z = (f_best - mu) / (sigma + eps)
    
    # Probability of Improvement: P(f(x) < f_best)
    # This is CDF of standard normal at z
    pi = norm.cdf(z)
    
    # Cost-aware score: Higher is better.
    # We want high probability of improvement and low cost.
    # So we divide by cost.
    # Ensure cost is not zero (it's >= 1.0 per problem description)
    scores = pi / (cost + eps)
    
    # Deadline guard check (strictly following directive, though unlikely to trigger)
    # If we were doing iterative refinement, we'd check here.
    # For a single vectorized pass, we just ensure we return finite values.
    # If time exceeded, we'd return a fallback, but 2000 points is trivial.
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
