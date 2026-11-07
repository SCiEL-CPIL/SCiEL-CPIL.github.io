"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 320  # 320s limit, 10s margin

def acquisition(mu, sigma, f_best, cost):
    # Check deadline
    if time.time() > _DEADLINE:
        return np.zeros_like(mu)
    
    # Linear Lower Confidence Bound (LBC) / UCB hybrid
    # Exploitation: f_best - mu (higher is better)
    # Exploration: beta * sigma (higher is better)
    # Cost normalization: divide by cost
    beta = 1.5  # Exploration weight
    score = (f_best - mu + beta * sigma) / (cost + 1e-9)
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    return score
# EVOLVE-BLOCK-END
