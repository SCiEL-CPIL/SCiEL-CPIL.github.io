"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    start_time = time.time()
    
    # UCB (Upper Confidence Bound) for minimization.
    # We want to minimize f(x).
    # We maximize the score.
    # A point is good if mu is low (predicted low value) and sigma is high (uncertainty).
    # Score = -mu + beta * sigma.
    # We divide by cost to be cost-aware.
    
    # Use a beta that balances exploration and exploitation.
    # Reduce beta slightly to favor exploitation more as we have a good baseline.
    beta = 0.7
    
    # Ensure sigma is non-negative and finite
    sigma = np.maximum(sigma, 1e-9)
    
    scores = (-mu + beta * sigma) / cost
    
    # Time guard
    if time.time() - start_time > 250:
        # Fallback: return a safe score, e.g., based only on mu
        scores = -mu / cost
        
    # Return finite scores
    return np.where(np.isfinite(scores), scores, 0.0)
# EVOLVE-BLOCK-END
