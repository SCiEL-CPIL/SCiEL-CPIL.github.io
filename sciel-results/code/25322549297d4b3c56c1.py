"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = 325.0  # 5s margin before 330s kill


def acquisition(mu, sigma, f_best):
    # Check deadline
    if time.time() > _DEADLINE:
        # Fallback: just return mu-based scores (lower mu = better)
        return -mu
    
    # Aggressive UCB with high exploration for multimodal Weierstrass
    # beta is scaled by the spread of the predictions to adapt to the problem scale
    beta = 2.5
    
    # UCB: mu - beta * sigma (we minimize, so lower is better)
    # But we want higher score = more promising
    # So score = -mu + beta * sigma
    scores = -mu + beta * sigma
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
