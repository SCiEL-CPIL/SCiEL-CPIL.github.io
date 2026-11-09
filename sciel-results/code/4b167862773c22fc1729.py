"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = None
acquisition._count = 0

def acquisition(mu, sigma, f_best):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Safety margin: ensure we return well before the 300s limit
    elapsed = time.time() - _start_time
    if elapsed > 280:
        # Fallback to pure EI to be safe, though this shouldn't happen often
        from scipy.stats import norm
        z = (f_best - mu) / (sigma + 1e-9)
        return (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)

    # Increment call counter
    acquisition._count += 1
    
    # Total steps = 30.
    # First 5 are random. So acquisition is called for steps 6..30.
    # That's 25 calls to acquisition.
    # Step 1 of acquisition corresponds to eval 6.
    # Step 25 of acquisition corresponds to eval 30.
    
    total_acq_calls = 25
    current_call = min(acquisition._count, total_acq_calls)
    
    # Strategy: Pure GP-LCB
    # score = -mu - kappa * sigma
    # We want to maximize this score (which corresponds to minimizing mu + kappa * sigma)
    
    # Fixed kappa as per directive
    kappa = 2.0
    
    # Compute LCB score
    # Higher score is better
    scores = -mu - kappa * sigma
    
    return scores
# EVOLVE-BLOCK-END
