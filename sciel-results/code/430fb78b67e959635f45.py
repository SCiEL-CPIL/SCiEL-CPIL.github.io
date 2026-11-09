"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state to track evaluation count and timing
_state = {
    'call_count': 0,
    'start_time': None
}

def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition function.
    
    Strategy:
    - Draws a single sample from the posterior predictive distribution at each candidate point.
    - Scores candidates by this sample.
    - This naturally balances exploration (high sigma leads to high variance in samples) 
      and exploitation (low mu leads to low samples) without requiring explicit weighting 
      or normalization across the batch.
    """
    # Initialize state
    if _state['start_time'] is None:
        _state['start_time'] = time.time()
    _state['call_count'] += 1
    
    # Deadline guard: If we are getting close to the limit, return a simple fallback.
    # However, Thompson Sampling is extremely fast (O(N) random draws), so this is likely unnecessary
    # but included for safety as per directive.
    elapsed = time.time() - _state['start_time']
    if elapsed > 320:
        # Fallback: return -mu (pure exploitation) to be safe
        return -mu

    # Draw a sample from N(mu, sigma^2)
    # mu and sigma are 1D arrays of shape (2000,)
    # We want to maximize the score, but we are minimizing the function.
    # So we should minimize the sample, or equivalently maximize -sample.
    # The contract says "higher meaning evaluate next".
    # So score = -sample.
    
    # Generate random noise
    eps = np.random.randn(len(mu))
    
    # Sample value
    sample = mu + sigma * eps
    
    # Score: higher is better, so we negate the sample (since lower function value is better)
    scores = -sample
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
