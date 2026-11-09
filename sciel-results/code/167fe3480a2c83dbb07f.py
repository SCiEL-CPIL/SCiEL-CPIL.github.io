"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state to track time and potentially seed if needed, though task says fresh uniform candidates each step
_start_time = None
_call_count = 0

def acquisition(mu, sigma, f_best):
    global _start_time, _call_count
    
    # Initialize time tracking if not already set
    if _start_time is None:
        _start_time = time.time()
        
    elapsed = time.time() - _start_time
    
    # Safety margin: return valid scores immediately if close to limit
    # The evaluator kills after 330s, task limit is 300s.
    # If we are past 290s, just return something valid (e.g., 0 or mu) to finish quickly.
    if elapsed > 290.0:
        # Return a neutral score to avoid crashing, effectively ending search or making a random pick
        # Since we need to return 2000 scores, we return zeros or mu.
        # Returning mu is safer than 0 if mu has scale, but 0 is fine as long as it's finite.
        return np.zeros_like(mu)

    # Increment call count to track progress (each call corresponds to one BO step)
    # The first 5 evaluations are initial random points, so the first 5 calls to acquisition
    # happen after those points are evaluated? No, the task says "starts from 5 uniform random points,
    # then at each step draws 2000...". So acquisition is called for steps 6 through 30.
    # That means acquisition is called 25 times total per seed.
    # However, since we don't know which seed we are in, and the state is global,
    # _call_count will accumulate across seeds.
    # Seed 0: calls 1-25
    # Seed 1: calls 26-50
    # Seed 2: calls 51-75
    # We can use this to infer the phase within a seed if we reset, but we can't reset easily.
    # Instead, we can just use the total call count to estimate the global phase,
    # or better, just use Thompson Sampling which doesn't require a tuned schedule.
    
    _call_count += 1

    # Thompson Sampling:
    # Sample y ~ N(mu, sigma^2)
    # We want to minimize y.
    # The interface asks for scores where HIGHER is better.
    # So we return score = -y.
    # This naturally balances exploration and exploitation.
    # High sigma regions have a higher chance of sampling a low value (good for minimization),
    # but also a chance of sampling a high value.
    # Low sigma regions will consistently sample values close to mu.
    # If mu is low, -y will be high.
    
    # Ensure sigma is non-negative and small enough to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Generate random samples
    # Use a fixed seed? No, random exploration is key for TS.
    # But we need to be deterministic for reproducibility? The task doesn't specify,
    # but standard practice is to use a global RNG.
    # However, if we use np.random.rand, it might be affected by other code.
    # Let's use a dedicated RNG or just np.random.
    
    # To ensure we don't hit the same sample every time if called rapidly,
    # but since each call is a new step, it should be fine.
    
    z = np.random.randn(mu.shape[0])
    y_samples = mu + sigma_safe * z
    
    # Score is negative of the sample, so higher score means lower predicted value
    scores = -y_samples
    
    # Ensure finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
