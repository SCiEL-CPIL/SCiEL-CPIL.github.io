"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()


def acquisition(mu, sigma, f_best):
    # Check time limit to ensure we return within constraints
    if time.time() - _start_time > 320:
        # Fallback: pure exploitation
        return -mu

    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-9)
    
    # Adaptive UCB parameter
    # In the early stages (few observations), we want more exploration.
    # As we collect more data, we can rely more on exploitation.
    # However, we don't have access to the number of observations directly.
    # We can infer the "stage" by looking at the magnitude of sigma or f_best.
    # But a simpler heuristic is to use a fixed, well-tuned beta.
    # The knowledge base suggests beta=2 improved performance over beta=1.
    # Let's try a slightly higher beta to encourage exploration in the multimodal landscape.
    beta = 2.0
    
    # Upper Confidence Bound for minimization
    # We want to maximize the score.
    # Score = -mu + beta * sigma
    # This balances finding low means (exploitation) with high uncertainty (exploration).
    scores = -mu + beta * sigma
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
