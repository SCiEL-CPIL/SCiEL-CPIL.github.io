import numpy as np
import time
from scipy.stats import norm

# Global state for deadline and step tracking
_start_time = None
_call_count = 0
# Estimated max calls: Initial 5 + ~20 more = 25. 
# Budget 30. Avg cost 1.5 -> 20 evals. 
# Let's assume max 25 steps for the adaptive schedule.
_MAX_STEPS = 25.0

def acquisition(mu, sigma, f_best, cost):
    global _start_time, _call_count
    
    # Initialize deadline tracker
    if _start_time is None:
        _start_time = time.time()
        
    # Increment call count
    _call_count += 1
    
    # Deadline check: 300s limit. Safety margin 20s.
    elapsed = time.time() - _start_time
    if elapsed > 280.0:
        # Fallback: Exploit best known location cheaply
        # Score based on low mu and low cost
        cost_safe = np.maximum(cost, 1e-6)
        scores = -mu / cost_safe
        scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
        return scores

    # Ensure inputs are numpy arrays
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    cost = np.asarray(cost)
    
    # Safe cost
    cost_safe = np.maximum(cost, 1.0) # Minimum cost is 1.0
    
    # Adaptive cost exponent gamma
    # Early on (low _call_count), gamma is low (0.5), allowing expensive points if they have high EI.
    # Late on (high _call_count), gamma is high (2.0), strongly penalizing expensive points.
    progress = min(_call_count / _MAX_STEPS, 1.0)
    gamma = 0.5 + 1.5 * progress
    
    # Calculate Expected Improvement (EI) for minimization
    # f_best is the current best (lowest) value.
    # Improvement = f_best - f(x).
    # E[Improvement] = (f_best - mu) * Phi(Z) + sigma * phi(Z)
    # where Z = (f_best - mu) / sigma
    
    # Handle sigma == 0 cases
    sigma_safe = np.maximum(sigma, 1e-8)
    
    Z = (f_best - mu) / sigma_safe
    
    # Compute Phi and phi
    # Use scipy for accuracy
    Phi = norm.cdf(Z)
    phi = norm.pdf(Z)
    
    EI = (f_best - mu) * Phi + sigma_safe * phi
    
    # EI can be negative if mu > f_best significantly, but mathematically EI >= 0 for standard EI definition?
    # Actually, if f_best - mu is negative, Phi is small, sigma*phi is positive.
    # Standard EI is non-negative.
    # Let's enforce non-negativity to be safe, though it shouldn't happen with correct math.
    EI = np.maximum(EI, 0.0)
    
    # Cost-aware score
    # Score = EI / (cost ^ gamma)
    scores = EI / (cost_safe ** gamma)
    
    # Ensure finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
