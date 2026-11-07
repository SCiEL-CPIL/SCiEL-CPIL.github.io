"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state to track evaluation count and start time
_call_count = 0
_start_time = time.time()

def acquisition(mu, sigma, f_best):
    global _call_count, _start_time
    
    # Deadline guard: The evaluator kills after 330s. 
    # This function is called 25 times (30 evals - 5 initial).
    # Each call is very fast, but we guard just in case.
    if time.time() - _start_time > 325:
        _call_count += 1
        # Fallback to simple exploitation
        return -mu

    _call_count += 1
    
    # t is the number of optimization steps taken so far (1-indexed)
    # Total optimization steps = 25
    t = _call_count
    
    # Adaptive LCB (Lower Confidence Bound) for minimization
    # We want to minimize f, so we look for low mu and high sigma.
    # Score = -mu + beta * sigma
    # High score = low mu (good) or high sigma (uncertain).
    
    # Decay beta from a high exploration value to a low exploitation value.
    # Start with beta=3.0 (strong exploration) and decay to ~0.5.
    # Using an exponential decay or power law.
    # Let's try a smooth decay that is high initially.
    # beta = 3.0 * exp(-0.1 * t) + 0.5
    # At t=1: 3*exp(-0.1)+0.5 ≈ 3*0.904+0.5 ≈ 3.2
    # At t=25: 3*exp(-2.5)+0.5 ≈ 3*0.082+0.5 ≈ 0.75
    
    # Alternatively, a power law decay might be more stable.
    # beta = 2.5 / (1 + 0.05 * t)
    # t=1: 2.5/1.05 ≈ 2.38
    # t=25: 2.5/2.25 ≈ 1.11
    
    # Let's use a slightly more aggressive start to ensure the first few
    # steps explore well, as EI often fails here.
    beta = 2.8 * np.exp(-0.08 * t) + 0.4
    
    # Calculate LCB score
    # mu and sigma are arrays of shape (2000,)
    scores = -mu + beta * sigma
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
