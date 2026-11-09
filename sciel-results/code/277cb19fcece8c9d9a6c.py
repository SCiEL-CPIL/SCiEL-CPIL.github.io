"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Precompute constants for speed
_EPS = 1e-12

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using a robust hybrid of Expected Improvement (EI) and
    a confidence-based exploration term. Instead of a portfolio search over
    hyperparameters (which is brittle without feedback), this uses a fixed,
    well-tuned blend of EI and UCB. EI drives exploitation toward known good
    regions, while a small UCB term ensures sufficient exploration in the
    early stages and prevents premature convergence.
    
    The blend is: score = EI + alpha * sigma
    where alpha is a small constant that balances exploration.
    
    We also handle numerical stability carefully.
    """
    start_time = time.time()
    
    # Compute EI with a small epsilon to encourage exploration slightly
    # Using xi=0.01 is a common robust choice
    xi = 0.01
    with np.errstate(divide='ignore', invalid='ignore'):
        z = (f_best - mu - xi) / (sigma + _EPS)
        ei = (f_best - mu - xi) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure EI is non-negative (clip small negative values from numerical noise)
    ei = np.maximum(ei, 0.0)
    
    # Add a small exploration term based on sigma
    # This effectively makes it EI + alpha * sigma, which is similar to
    # a soft UCB but anchored to the improvement expectation
    alpha = 0.1
    scores = ei + alpha * sigma
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
