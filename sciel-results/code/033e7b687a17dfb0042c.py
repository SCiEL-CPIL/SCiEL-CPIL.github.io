# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state to persist across calls within the same seed/evaluation run
# We use a simple counter to vary the random seed slightly if needed, 
# though standard np.random is fine. 
# Actually, for Thompson Sampling, we just need to draw a sample each time.
# No complex state is strictly required, but we might want to ensure 
# we don't have issues with global RNG state if the evaluator runs in parallel?
# The task implies a single loop per seed. Standard numpy random is acceptable.

def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition function for Bayesian Optimization.
    
    Parameters:
    mu: array-like, shape (N,) - GP posterior mean
    sigma: array-like, shape (N,) - GP posterior standard deviation
    f_best: float - Best observed value so far
    
    Returns:
    scores: array, shape (N,) - Scores where higher is better to select.
    """
    t_start = time.time()
    # Safety deadline guard: 300s limit, keep margin.
    deadline = t_start + 280.0
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    n = len(mu)
    if n == 0:
        return np.array([], dtype=np.float64)
    
    # Ensure sigma is non-negative and handles zeros safely
    # In TS, if sigma is 0, the sample is just mu.
    # If sigma is very small, the sample is close to mu.
    # We can just use sigma directly.
    
    # Thompson Sampling:
    # Draw a sample f_sample ~ N(mu, sigma^2)
    # For minimization, we want to select the point with the lowest sampled value.
    # The task requires returning scores where HIGHER means evaluate next.
    # So we return -f_sample.
    
    # Vectorized draw: z ~ N(0, 1)
    z = np.random.standard_normal(n)
    
    # f_sample = mu + sigma * z
    # Score = -f_sample = -(mu + sigma * z) = -mu - sigma * z
    
    scores = -mu - sigma * z
    
    # Ensure finite values
    # If sigma is huge, scores could be -inf? 
    # mu and sigma are from GP, typically finite. 
    # But let's be safe.
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e10, neginf=-1e10)
    
    # Check time
    if time.time() > deadline:
        # This is extremely fast, so this should never trigger.
        # If it does, return something deterministic to avoid crash?
        # Just return the scores we have.
        pass
        
    return scores
# EVOLVE-BLOCK-END
