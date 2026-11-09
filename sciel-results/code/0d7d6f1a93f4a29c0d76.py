import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE_MARGIN = 5.0  # seconds to leave before the 330s limit

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: if we're close to the 330s wall-clock limit, return quickly
    elapsed = time.time() - _start_time
    if elapsed > 330.0 - _DEADLINE_MARGIN:
        # Return a safe fallback: uniform random scores to avoid invalid output
        n = len(mu)
        return np.random.uniform(0, 1, size=n)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    eps = 1e-9
    
    # Expected Improvement
    z = (f_best - mu) / (sigma + eps)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Add a mild exploration term: boost points with high uncertainty
    # This helps explore before committing to exploitation
    exploration_weight = 0.1
    exploration = exploration_weight * sigma
    
    # Combine EI and exploration
    numerator = ei + exploration
    
    # Cost-aware penalty: divide by cost to prefer cheaper evaluations
    # Use a small offset to avoid division by zero
    scores = numerator / (cost + eps)
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
