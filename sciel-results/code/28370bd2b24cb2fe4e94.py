"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Deadline guard: ensure we don't exceed the time limit
    # The computation is very fast, but we include a guard as per directive.
    start_time = time.time()
    
    # Thompson Sampling Strategy:
    # 1. Sample a value for each candidate from the posterior distribution N(mu, sigma)
    # 2. The candidate with the lowest sampled value is considered the best next evaluation
    # 3. We return negative sampled values so that the maximum score corresponds to the minimum sampled value
    # This stochastic approach balances exploration and exploitation more robustly than EI
    # in multi-modal landscapes like Weierstrass.
    
    samples = np.random.normal(mu, sigma)
    
    # Safety check: if the operation took unexpectedly long (e.g., due to system load),
    # fall back to a deterministic method to ensure we return in time.
    if time.time() - start_time > 290.0:
         z = (f_best - mu) / (sigma + 1e-9)
         return (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
         
    return -samples
# EVOLVE-BLOCK-END
