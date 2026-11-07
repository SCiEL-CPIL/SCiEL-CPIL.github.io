import numpy as np
from scipy.stats import norm

# Global state to track optimization progress
_call_count = 0

def acquisition(mu, sigma, f_best):
    global _call_count
    
    # 1. Increment step counter
    _call_count += 1
    
    # 2. Determine step progress
    # Total evaluations = 30. Initial 5 are random. So 25 acquisition calls.
    max_steps = 25
    if _call_count > max_steps:
        _call_count = max_steps
        
    progress = _call_count / max_steps
    
    # 3. Compute UCB parameter beta
    # UCB = mu - beta * sigma (since we are minimizing)
    # We want to explore more in the beginning and exploit more later.
    # A common heuristic is to decay beta.
    # Let's try a decay from a high value to a low value.
    # Ackley 2D range is approx [0, 7].
    # If beta is too high, we pick the most uncertain point (often far from f_best).
    # If beta is low, we pick the lowest mu (exploit).
    
    # Linear decay of beta
    beta_start = 2.0
    beta_end = 0.5
    beta = beta_start + (beta_end - beta_start) * progress
    
    # 4. Compute UCB scores
    # Ensure sigma is non-negative and finite
    sigma_safe = np.maximum(sigma, 1e-8)
    
    ucb = mu - beta * sigma_safe
    
    # 5. Safety checks
    scores = np.where(np.isfinite(ucb), ucb, -np.inf)
    
    # If all scores are -inf, return zeros to avoid crash
    if np.all(scores == -np.inf):
        return np.zeros_like(mu)
        
    return scores
