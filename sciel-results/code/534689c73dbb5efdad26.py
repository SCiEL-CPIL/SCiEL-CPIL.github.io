import numpy as np
import time

_start_time = None
_call_count = 0
# Use a local RNG instance to avoid interfering with global state if possible, 
# though global is usually fine. Let's use np.random default for simplicity 
# unless reproducibility issues arise. 
# Actually, creating a new Generator every call is expensive. 
# Let's stick to np.random.randn.

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Thompson Sampling acquisition function.
    
    Args:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP standard deviation predictions
        f_best: float - best observed function value so far
        cost: array of shape (2000,) - evaluation costs for each candidate
    
    Returns:
        array of shape (2000,) - acquisition scores (higher = better)
    """
    global _start_time, _call_count
    
    if _start_time is None:
        _start_time = time.time()
    _call_count += 1
    
    elapsed = time.time() - _start_time
    if elapsed > 325:
        # Safety: return exploit-heavy fallback
        cost_safe = np.maximum(cost, 1e-9)
        # Penalize cost, prefer low mu
        return -mu / (cost_safe + 1e-9)
    
    # Clip sigma to avoid numerical issues
    sigma = np.clip(sigma, 1e-9, 1e6)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Generate standard normal random numbers for Thompson Sampling
    z = np.random.randn(2000)
    
    # Sample function values from the posterior
    # For minimization, we sample y ~ N(mu, sigma^2)
    # A "good" sample is one that is lower than f_best
    samples = mu - sigma * z
    
    # Calculate instantaneous improvement
    # Improvement is positive if the sample beats the current best
    improvement = np.maximum(0.0, f_best - samples)
    
    # Cost-aware score: Improvement per unit cost
    base_score = improvement / cost_safe
    
    # Add an uncertainty bonus to encourage exploration
    # This helps when the posterior is confident (low sigma) but might be wrong,
    # or when improvements are rare.
    if _call_count <= 5:
        exploration_factor = 0.15
    elif _call_count <= 10:
        exploration_factor = 0.10
    elif _call_count <= 20:
        exploration_factor = 0.05
    elif _call_count <= 40:
        exploration_factor = 0.02
    else:
        exploration_factor = 0.01
        
    exploration_bonus = exploration_factor * (sigma / cost_safe)
    
    scores = base_score + exploration_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
