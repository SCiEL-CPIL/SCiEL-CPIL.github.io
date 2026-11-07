"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Upper Confidence Bound for minimization.
    # We want to minimize f(x).
    # The lower confidence bound is L(x) = mu(x) - beta * sigma(x).
    # We want to pick the x with the lowest L(x).
    # Therefore, we maximize the score A(x) = -L(x) = -mu(x) + beta * sigma(x).
    
    beta = 1.5
    
    # Ensure sigma is positive to avoid numerical issues, though GP sigma should be > 0
    # Adding a small epsilon for safety
    sigma_safe = sigma + 1e-9
    
    scores = -mu + beta * sigma_safe
    
    return scores
# EVOLVE-BLOCK-END
