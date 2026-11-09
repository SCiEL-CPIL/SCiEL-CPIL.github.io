import numpy as np
import time

# Global state to track steps and time
_start_time = None
_step_counter = 0
_last_f_best = None

def acquisition(mu, sigma, f_best):
    global _start_time, _step_counter, _last_f_best
    
    if _start_time is None:
        _start_time = time.time()
    
    # Detect new seed/run
    # If f_best is significantly higher than the last observed min, assume new run
    if _last_f_best is not None and f_best > _last_f_best + 0.5:
        _step_counter = 0
    else:
        _step_counter += 1
        
    _last_f_best = f_best
    
    # Deadline guard: if we are running low on time, return a simple heuristic
    # Though TS is very fast, we keep this for safety.
    elapsed = time.time() - _start_time
    if elapsed > 320:
        # Fallback to greedy (min mu) if something goes wrong or time is tight
        return -mu

    # Thompson Sampling
    # We sample a value from the posterior distribution N(mu, sigma^2) for each candidate.
    # Since we are minimizing, the candidate with the lowest sampled value is the best.
    # The function requires scores where HIGHER is better.
    # So we return -sampled_value.
    
    # Ensure sigma is positive to avoid issues with random generation
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Sample from N(mu, sigma_safe)
    # np.random.normal takes loc and scale
    # We use a fixed seed per step? No, standard TS uses random sampling.
    # However, to ensure reproducibility or stability in some evaluators, 
    # sometimes a deterministic seed is preferred, but the task implies a stochastic process.
    # The prompt doesn't forbid randomness.
    
    samples = np.random.normal(loc=mu, scale=sigma_safe)
    
    # We want the candidate with the minimum sample to have the highest score.
    # Score = -sample
    scores = -samples
    
    # Optional: Add a small exploitation bias to the samples?
    # Pure TS is often sufficient. 
    # However, if the GP is very confident (low sigma), TS becomes greedy.
    # If the GP is uncertain, it explores.
    
    return scores
