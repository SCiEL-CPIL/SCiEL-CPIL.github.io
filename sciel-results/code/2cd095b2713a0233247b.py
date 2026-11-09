"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state for Thompson Sampling
_thompson_state = {
    'last_call_time': 0.0,
    'last_f_best': None,
    'sample': None
}

def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition function.
    
    Strategy:
    - Draw a single sample from the posterior predictive distribution N(mu, sigma^2).
    - To ensure stability and avoid redundant computations within the same 'step' 
      (if called multiple times with identical inputs, though the contract implies 
      one call per step with 2000 candidates), we use a deterministic pseudo-random 
      seed based on the call count or time to ensure reproducibility if needed, 
      but here we just use a standard normal draw.
    - The score for each candidate is the sampled value.
    - Minimizing the sampled value is equivalent to maximizing -sampled value.
    - Since the task requires "higher meaning evaluate next", we return -sampled value.
    
    Wait, the task says "minimizes hartmann 6d". The evaluator measures raw value (lower is better).
    The acquisition function should return scores where higher is better for the *selection* of the next point.
    So if we sample y ~ N(mu, sigma^2), and we want to find the minimum, we should pick the x that minimizes y.
    Therefore, the score should be -y.
    """
    current_time = time.time()
    
    # We need to be careful about the deadline.
    # The problem states: "Keep an EXPLICIT deadline guard: track elapsed time and return the best result found with a safety margin before the limit."
    # However, the acquisition function itself is very fast (O(N) for N=2000).
    # The 300s limit applies to the *entire* evaluation loop per seed or total? 
    # "Each evaluation must finish within 300 s." usually means the whole script execution for one seed.
    # Since we don't control the loop, we just need to ensure this function is fast.
    # Thompson sampling is O(N) for 2000 points, which is negligible.
    
    # Draw a single standard normal random number
    # Using numpy's global random state is fine, but for reproducibility in tests, 
    # sometimes fixed seeds are preferred. However, TS requires randomness.
    # The prompt implies we are in a loop. 
    # To make it deterministic for the same input sequence (if the evaluator re-runs), 
    # we might want to seed based on something, but standard TS uses fresh randomness.
    # Let's use np.random.normal.
    
    z = np.random.normal(0.0, 1.0)
    
    # Sampled objective value: y = mu + sigma * z
    # We want to minimize y, so we maximize -y
    scores = -(mu + sigma * z)
    
    # Ensure finite scores
    # If sigma or mu are inf/nan, handle it. But GP outputs should be finite.
    # Just in case:
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
