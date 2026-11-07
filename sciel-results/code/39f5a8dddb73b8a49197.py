"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_deadline = _start_time + 290.0
_call_count = 0

def acquisition(mu, sigma, f_best):
    global _call_count
    
    # Deadline guard: if we are running out of time, fall back to greedy
    if time.time() > _deadline:
        return -mu

    _call_count += 1
    
    # Ensure sigma is strictly positive to avoid numerical issues
    sigma = np.maximum(sigma, 1e-9)
    
    # Thompson Sampling: sample from the predictive posterior
    # For each candidate i, sample f_i ~ N(mu_i, sigma_i^2)
    # Score is -f_i (higher means lower predicted value, so we want to evaluate)
    
    # Generate random samples for Thompson Sampling
    # Use a deterministic seed based on call count for reproducibility within a run
    rng = np.random.default_rng(seed=_call_count * 7919 + 42)
    
    # Draw samples: s_i = mu_i + sigma_i * z_i where z_i ~ N(0, 1)
    z_samples = rng.standard_normal(size=mu.shape)
    sampled_values = mu + sigma * z_samples
    
    # Score: we want to minimize, so higher score = better (lower value)
    # Use negative of sampled value so that lower objective -> higher score
    scores = -sampled_values
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
