import numpy as np
import time

def acquisition(mu, sigma, f_best):
    start_time = time.time()
    # Explicit deadline guard: 290s limit to ensure we return before 300s
    deadline = start_time + 290.0
    
    sigma = np.maximum(sigma, 1e-8)
    
    # Posterior sampling: 50 samples
    n_samples = 50
    z = np.random.standard_normal((n_samples, mu.shape[0]))
    f_samples = mu[None, :] + sigma[None, :] * z
    
    # Calculate improvements for each sample
    improvements = np.maximum(0, f_best - f_samples)
    
    # Check deadline (computation is fast, but guard is required)
    if time.time() > deadline:
        # Fallback to Mean Improvement (EI) if time runs out
        return np.mean(improvements, axis=0)
        
    # Strategy: Take the maximum improvement across samples
    # This replaces the deterministic UCB heuristic
    scores = np.max(improvements, axis=0)
    
    return scores
